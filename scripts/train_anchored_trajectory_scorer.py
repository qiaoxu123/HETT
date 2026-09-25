#!/usr/bin/env python3
"""Rank eight anchored navigation-and-observation trajectories.

This stage reuses ``TrajectoryHypothesisScorer``.  Ground-truth targets are
used only to produce ranking labels and metrics; route generation and scorer
forward inputs remain target-free.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import belief_distinct_topk_xy
from independent_nav.trajectory_library import generate_rule_trajectory_set
from independent_nav.trajectory_scorer import MultiObjectiveTrajectoryScorer
from independent_nav.trajectory_utility import geometric_trajectory_utility
from scripts.build_temporal_belief_dataset import pose_xy_yaw
from scripts.evaluate_official_static_belief import (
    DEFAULT_SOURCE, BeliefInferenceDataset, load_rows, predict_beliefs,
)
from scripts.train_relation_geometry_ranker import build_split, ranking_loss
from scripts.train_rgb_candidate_ranker import metrics
from scripts.train_trajectory_hypothesis_scorer import (
    belief_features, normalized_start,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--belief-checkpoint", type=Path, default=Path(
        "runs/belief_transformer_stage1_relative/model_train3epoch.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/anchored_trajectory_scorer_train1024_holdout256"))
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--holdout-count", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hit-loss-weight", type=float, default=0.2)
    parser.add_argument("--utility-reward-m", type=float, default=20.0)
    parser.add_argument("--utility-loss-weight", type=float, default=1.0)
    parser.add_argument("--utility-score-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


SLOT_TO_MODE = (0, 1, 2, 3, 4, 0, 0, 1)


def build_features(args, relation_all, start, count, device):
    stop = start + count
    rows = load_rows(args.candidate_dir / "train_seen.jsonl")[start:stop]
    trajectories = json.loads((args.data_root / "processed_citynav" /
                               "citynav_train_seen.json").read_text())[start:stop]
    relation = relation_all[start:stop]
    inference = BeliefInferenceDataset(rows, relation)
    beliefs, _ = predict_beliefs(
        args.belief_checkpoint, inference, device, args.batch_size)
    modes = [belief_distinct_topk_xy(value, 5, 25.0) for value in beliefs]
    enriched_rows, paths, path_positions, path_valids, endpoints = [], [], [], [], []
    for index, (row, trajectory, hypotheses) in enumerate(
            zip(rows, trajectories, modes)):
        map_name = f'{trajectory["area"]}_block_{trajectory["block"]}'
        if map_name != row["map_name"] or trajectory["descriptions"][0] != row["description"]:
            raise ValueError(f"row alignment mismatch at train_seen:{start + index}")
        start_xy = normalized_start(trajectory["trajectory"][0], map_name)
        _, _, yaw = pose_xy_yaw(trajectory["trajectory"][0])
        library = generate_rule_trajectory_set(start_xy, yaw, hypotheses)
        endpoint = [value.positions[-1].tolist() for value in library]
        value = copy.copy(row); value["candidates"] = endpoint
        enriched_rows.append(value); endpoints.append(endpoint)
        action = torch.stack([value.actions for value in library])
        valid = torch.stack([value.valid for value in library])
        position = torch.stack([value.positions[1:] for value in library])
        standardized = (beliefs[index] - beliefs[index].mean()) / beliefs[index].std().clamp_min(1e-5)
        grid = standardized.shape[-1]
        column = (position[..., 0] * grid).long().clamp(0, grid - 1)
        row_index = (position[..., 1] * grid).long().clamp(0, grid - 1)
        path_belief = standardized[row_index, column][..., None]
        # Absolute world path and sampled belief make this a whole-trajectory
        # representation rather than another endpoint-only candidate scorer.
        paths.append(torch.cat((
            action, position, path_belief, valid[..., None].float()), -1))
        path_positions.append(position); path_valids.append(valid)
    data = build_split(enriched_rows, relation, 9)
    belief = belief_features(beliefs, endpoints)
    for slot, mode in enumerate(SLOT_TO_MODE):
        belief[:, slot, 3] = -mode / 4.0
    data.update({
        "trajectory": torch.stack(paths),
        "belief": belief,
        "endpoints": torch.tensor(endpoints),
    })
    teacher = geometric_trajectory_utility(
        torch.stack(path_positions), torch.stack(path_valids),
        data["endpoints"], inference.centers, inference.valid,
        torch.tensor(modes))
    data.update(teacher)
    data["training_distance"] = (
        data["distance"] - args.utility_reward_m * data["utility"])
    return data


def scorer_dataset(data):
    return TensorDataset(
        data["relation"], data["cue"], data["geometry"], data["valid"],
        data["trajectory"], data["belief"], data["distance"], data["utility"])


@torch.inference_mode()
def predict(model, data, device, ablation=None):
    values = [data[key] for key in (
        "relation", "cue", "geometry", "valid", "trajectory", "belief")]
    if ablation == "no_route":
        values[4] = values[4].clone(); values[4][..., :-1] = 0
    elif ablation == "no_belief":
        values[5] = torch.zeros_like(values[5])
    output = []
    loader = DataLoader(TensorDataset(*values), batch_size=256)
    for batch in loader:
        output.append(model(*(item.to(device) for item in batch)).cpu())
    return torch.cat(output)


def evaluate(model, data, device):
    score = predict(model, data, device)
    no_route_score = predict(model, data, device, "no_route")
    no_belief_score = predict(model, data, device, "no_belief")
    distance = data["distance"]
    selected = score.argmax(-1)
    no_route_selected = no_route_score.argmax(-1)
    no_belief_selected = no_belief_score.argmax(-1)
    return {
        "selected": metrics(distance, selected),
        "belief_top1": metrics(
            distance, torch.zeros(len(distance), dtype=torch.long)),
        "best_of_8_oracle": metrics(distance, distance.argmin(-1)),
        "no_route": metrics(distance, no_route_selected),
        "no_belief": metrics(distance, no_belief_selected),
        "selection_histogram": {
            str(index): int(selected.eq(index).sum()) for index in range(8)},
        "selected_teacher_utility": float(
            data["utility"].gather(1, selected[:, None]).mean()),
        "no_route_selected_teacher_utility": float(
            data["utility"].gather(1, no_route_selected[:, None]).mean()),
        "no_belief_selected_teacher_utility": float(
            data["utility"].gather(1, no_belief_selected[:, None]).mean()),
        "top1_teacher_utility": float(data["utility"][:, 0].mean()),
        "oracle_teacher_utility": float(data["utility"].amax(-1).mean()),
    }, score


def train(train_data, holdout, args, device):
    model = MultiObjectiveTrajectoryScorer(
        route_channels=8, utility_scale=args.utility_score_scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    best_key, best_state, best_epoch, history = None, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        loader = DataLoader(scorer_dataset(train_data), batch_size=args.batch_size,
                            shuffle=True, pin_memory=True)
        for batch in loader:
            relation, cue, geometry, valid, trajectory, belief, distance, utility = (
                value.to(device, non_blocking=True) for value in batch)
            output = model(
                relation, cue, geometry, valid, trajectory, belief,
                return_components=True)
            goal_loss = ranking_loss(
                output["goal"], distance, args.hit_loss_weight)[0]
            final_loss = ranking_loss(
                output["final"], distance - args.utility_reward_m * utility,
                args.hit_loss_weight)[0]
            utility_loss = torch.nn.functional.mse_loss(
                output["utility"], utility)
            loss = goal_loss + final_loss + args.utility_loss_weight * utility_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); result, _ = evaluate(model, holdout, device)
        selected = result["selected"]
        key = (selected["hit20"], -selected["mean_ne"])
        record = {"epoch": epoch, "loss": float(np.mean(losses)),
                  "holdout": result}
        history.append(record)
        print(f"epoch={epoch}/{args.epochs} loss={record['loss']:.4f} "
              f"hit20={selected['hit20']:.2f} ne={selected['mean_ne']:.2f}",
              flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), best_epoch, history


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relation = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True)["train_seen"]
    train_data = build_features(args, relation, 0, args.train_count, device)
    holdout = build_features(
        args, relation, args.train_count, args.holdout_count, device)
    model, best_epoch, history = train(train_data, holdout, args, device)
    train_result, _ = evaluate(model, train_data, device)
    holdout_result, score = evaluate(model, holdout, device)
    result = {
        "experiment": "eight anchored whole-trajectory ranking",
        "best_epoch": best_epoch,
        "train": train_result, "holdout": holdout_result,
        "history": history,
        "anti_leakage": {
            "target_in_route_generation": False,
            "target_in_scorer_forward": False,
            "target_use": "ranking labels and final metrics only",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
        },
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "best_epoch": best_epoch,
                "config": result["config"]}, args.output_dir / "model.pt")
    torch.save({"score": score, "selected": score.argmax(-1),
                "endpoints": holdout["endpoints"]},
               args.output_dir / "holdout_selection.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(holdout_result, indent=2))


if __name__ == "__main__":
    main()
