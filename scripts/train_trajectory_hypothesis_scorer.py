#!/usr/bin/env python3
"""Train a target-free scorer for Top-K deterministic navigation paths."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import (
    OfficialProtocol, belief_distinct_topk_xy,
)
from independent_nav.trajectory_scorer import TrajectoryHypothesisScorer
from multiagent.mapdata import MAP_BOUNDS
from scripts.build_temporal_belief_dataset import pose_xy_yaw
from scripts.evaluate_official_static_belief import (
    DEFAULT_SOURCE, BeliefInferenceDataset, load_rows, predict_beliefs,
)
from scripts.train_relation_geometry_ranker import (
    build_split, listwise_loss_per_row, ranking_loss,
)
from scripts.train_rgb_candidate_ranker import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--belief-checkpoint", type=Path, default=Path(
        "runs/belief_transformer_stage1_relative/model_train3epoch.pt"))
    parser.add_argument("--cache", type=Path, default=Path(
        "runs/trajectory_hypothesis_scorer/features.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/trajectory_hypothesis_scorer"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-separation-m", type=float, default=25.0)
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hit-loss-weight", type=float, default=0.2)
    parser.add_argument("--mismatch-loss-weight", type=float, default=0.2)
    parser.add_argument("--mismatch-margin", type=float, default=0.2)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def normalized_start(raw_pose, map_name):
    x, y, _ = pose_xy_yaw(raw_pose)
    bounds = MAP_BOUNDS[map_name]
    return ((x - bounds.x_min) / 410.0, (bounds.y_max - y) / 410.0)


def route_tensor(starts, modes, max_actions, protocol):
    output = torch.zeros(len(starts), len(modes[0]), max_actions, 5)
    for row_index, (start, hypotheses) in enumerate(zip(starts, modes)):
        start = np.asarray(start, dtype=np.float32)
        for mode_index, mode in enumerate(hypotheses):
            delta = np.asarray(mode, dtype=np.float32) - start
            distance_m = float(np.linalg.norm(delta) * protocol.map_size_m)
            movements = min(max_actions, max(
                1, int(math.ceil(distance_m /
                                 protocol.maximum_translation_per_action_m))))
            step = delta / movements
            yaw = math.atan2(-float(delta[1]), float(delta[0]))
            output[row_index, mode_index, :movements, 0] = float(
                step[0] * protocol.map_size_m
                / protocol.maximum_translation_per_action_m)
            output[row_index, mode_index, :movements, 1] = float(
                step[1] * protocol.map_size_m
                / protocol.maximum_translation_per_action_m)
            output[row_index, mode_index, :movements, 2] = math.sin(yaw)
            output[row_index, mode_index, :movements, 3] = math.cos(yaw)
            output[row_index, mode_index, :movements, 4] = 1.0
    return output


def belief_features(beliefs, modes):
    batch, grid, _ = beliefs.shape
    probability = beliefs.flatten(1).softmax(-1).reshape_as(beliefs)
    mean = beliefs.flatten(1).mean(-1)
    scale = beliefs.flatten(1).std(-1).clamp_min(1e-5)
    result = torch.zeros(batch, len(modes[0]), 4)
    for row_index, hypotheses in enumerate(modes):
        for rank, (x, y) in enumerate(hypotheses):
            column = min(grid - 1, max(0, int(float(x) * grid)))
            row = min(grid - 1, max(0, int(float(y) * grid)))
            y0, y1 = max(0, row - 1), min(grid, row + 2)
            x0, x1 = max(0, column - 1), min(grid, column + 2)
            result[row_index, rank] = torch.tensor((
                float((beliefs[row_index, row, column] - mean[row_index])
                      / scale[row_index]),
                float(probability[row_index, row, column].clamp_min(1e-8).log()
                      / 10.0),
                float(probability[row_index, y0:y1, x0:x1].sum()),
                -rank / max(len(hypotheses) - 1, 1),
            ))
    return result


def build_features(args, split, relation, device, limit):
    rows = load_rows(args.candidate_dir / f"{split}.jsonl")
    trajectories = json.loads((args.data_root / "processed_citynav" /
                               f"citynav_{split}.json").read_text())
    if limit:
        rows, trajectories, relation = rows[:limit], trajectories[:limit], relation[:limit]
    inference = BeliefInferenceDataset(rows, relation)
    beliefs, _ = predict_beliefs(
        args.belief_checkpoint, inference, device, args.batch_size)
    modes = [belief_distinct_topk_xy(
        value, args.top_k, args.minimum_separation_m) for value in beliefs]
    mode_rows, starts = [], []
    for index, (row, trajectory, hypotheses) in enumerate(
            zip(rows, trajectories, modes)):
        name = f'{trajectory["area"]}_block_{trajectory["block"]}'
        if name != row["map_name"] or trajectory["descriptions"][0] != row["description"]:
            raise ValueError(f"row alignment mismatch at {split}:{index}")
        value = copy.copy(row)
        value["candidates"] = [list(point) for point in hypotheses]
        mode_rows.append(value)
        starts.append(normalized_start(trajectory["trajectory"][0], name))
    data = build_split(mode_rows, relation, 9)
    data.update({
        "trajectory": route_tensor(
            starts, modes, args.max_actions, OfficialProtocol()),
        "belief": belief_features(beliefs, modes),
        "modes": torch.tensor(modes),
        "start": torch.tensor(starts),
    })
    return data


def load_or_build(args, device):
    if args.cache.exists():
        return torch.load(args.cache, map_location="cpu", weights_only=True)
    relation = torch.load(args.relation_cache, map_location="cpu", weights_only=True)
    saved = {
        "train_seen": build_features(
            args, "train_seen", relation["train_seen"], device, args.max_train),
        "val_seen": build_features(
            args, "val_seen", relation["val_seen"], device, args.max_val),
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.cache)
    return saved


def dataset(data):
    return TensorDataset(
        data["relation"], data["cue"], data["geometry"], data["valid"],
        data["trajectory"], data["belief"], data["distance"],
        data["opposite_cue"], data["switched_relation"],
        data["switched_cue"], data["switched_geometry"],
        data["switched_valid"], data["switch_changed"])


@torch.inference_mode()
def predict(model, data, device, control="normal"):
    relation, cue = data["relation"], data["cue"]
    geometry, valid = data["geometry"], data["valid"]
    trajectory, belief = data["trajectory"], data["belief"]
    if control == "opposite_relation":
        cue = data["opposite_cue"]
    elif control == "wrong_relation":
        relation, cue = data["switched_relation"], data["switched_cue"]
    elif control == "wrong_landmark":
        geometry, valid = data["switched_geometry"], data["switched_valid"]
    elif control == "no_belief":
        belief = torch.zeros_like(belief)
    elif control == "no_route":
        trajectory = trajectory.clone(); trajectory[..., :4] = 0
    values = TensorDataset(
        relation, cue, geometry, valid, trajectory, belief)
    output = []
    for batch in DataLoader(values, batch_size=256):
        output.append(model(*(value.to(device) for value in batch)).cpu())
    return torch.cat(output)


def evaluate(model, data, device):
    distance = data["distance"]
    names = ("normal", "opposite_relation", "wrong_relation",
             "wrong_landmark", "no_belief", "no_route")
    scores = {name: predict(model, data, device, name) for name in names}
    result = {name: metrics(distance, value.argmax(-1))
              for name, value in scores.items()}
    result.update({
        "belief_top1": metrics(
            distance, torch.zeros(len(distance), dtype=torch.long)),
        "top5_oracle": metrics(distance, distance.argmin(-1)),
        "selection_histogram": {
            str(index + 1): int(scores["normal"].argmax(-1).eq(index).sum())
            for index in range(distance.shape[1])},
    })
    return result, scores["normal"]


def train(train, val, args, device):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = TrajectoryHypothesisScorer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    best, best_state, best_epoch, history = None, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in DataLoader(
                dataset(train), batch_size=args.batch_size, shuffle=True,
                pin_memory=True):
            values = [value.to(device, non_blocking=True) for value in batch]
            normal = model(*values[:6])
            loss = ranking_loss(
                normal, values[6], args.hit_loss_weight)[0]
            if args.mismatch_loss_weight and values[12].any():
                opposite = model(
                    values[0], values[7], *values[2:6])
                wrong_relation = model(
                    values[8], values[9], *values[2:6])
                wrong_landmark = model(
                    values[0], values[1], values[10], values[11],
                    values[4], values[5])
                correct_ce = listwise_loss_per_row(normal, values[6])
                selected = values[12]
                mismatch = 0
                for wrong in (opposite, wrong_relation, wrong_landmark):
                    wrong_ce = listwise_loss_per_row(wrong, values[6])
                    mismatch = mismatch + F.softplus(
                        args.mismatch_margin + correct_ce[selected]
                        - wrong_ce[selected]).mean()
                loss = loss + args.mismatch_loss_weight * mismatch / 3
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); result, _ = evaluate(model, val, device)
        normal = result["normal"]
        baseline = result["belief_top1"]["hit20"]
        control_gap = min(
            normal["hit20"] - result[name]["hit20"] for name in (
                "opposite_relation", "wrong_relation", "wrong_landmark"))
        passes_control_gate = (
            normal["hit20"] >= baseline and control_gap >= 1.0)
        # A slightly better scorer that ignores the instruction is not a
        # valid checkpoint. Prefer checkpoints that first pass the grounding
        # gate, then maximize deployable accuracy.
        key = (int(passes_control_gate), normal["hit20"], control_gap,
               -normal["mean_ne"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "val_seen": result})
        print(f"epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} "
              f"hit20={normal['hit20']:.2f} mean_ne={normal['mean_ne']:.2f} "
              f"wrong_relation={result['wrong_relation']['hit20']:.2f} "
              f"wrong_landmark={result['wrong_landmark']['hit20']:.2f} "
              f"control_gap={control_gap:.2f} gate={passes_control_gate}",
              flush=True)
        if best is None or key > best:
            best, best_epoch = key, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), best_epoch, history


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = load_or_build(args, device)
    model, epoch, history = train(
        saved["train_seen"], saved["val_seen"], args, device)
    train_result, _ = evaluate(model, saved["train_seen"], device)
    val_result, val_score = evaluate(model, saved["val_seen"], device)
    result = {
        "stage": "target-free language/landmark/belief trajectory scorer",
        "selected_epoch_on_val_seen": epoch,
        "counts": {key: len(value["distance"]) for key, value in saved.items()},
        "train_seen": train_result,
        "val_seen": val_result,
        "selection_gate": {
            "minimum_wrong_input_drop_pp": 1.0,
            "must_match_or_beat_belief_top1": True,
            "passed": (
                val_result["normal"]["hit20"]
                >= val_result["belief_top1"]["hit20"]
                and min(val_result["normal"]["hit20"]
                        - val_result[name]["hit20"] for name in (
                            "opposite_relation", "wrong_relation",
                            "wrong_landmark")) >= 1.0),
        },
        "history": history,
        "anti_leakage": {
            "trajectory_generation_target_input": False,
            "scorer_forward_target_input": False,
            "target_usage": "training ranking labels and final metrics only",
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
        },
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "selected_epoch": epoch,
                "config": result["config"]}, args.output_dir / "model.pt")
    torch.save({"score": val_score,
                "selected": val_score.argmax(-1),
                "modes": saved["val_seen"]["modes"]},
               args.output_dir / "val_seen_selection.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["val_seen"], indent=2))


if __name__ == "__main__":
    main()
