#!/usr/bin/env python3
"""Train a visual-evidence stop head on official path-local trajectories.

Reference target coordinates are used only to construct the supervised
``within 20 m`` label.  The feature function and deployable model never receive
them.  Training uses train_seen; val_seen is used for checkpoint/threshold
selection.  No unseen split is loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.evidence_stop import (
    FEATURE_NAMES, EvidenceStopHead, center_semantic_statistics,
    first_stop_index, local_belief_statistics, stop_features,
)
from independent_nav.official_protocol import OfficialProtocol, move_toward
from independent_nav.semantic import ProjectSemanticNet
from multiagent.cityreferobject import get_city_refer_objects
from multiagent.dataset.generate import generate_episodes_from_mturk_trajectories
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.env import CityNavBatch
from multiagent.observation import cropclient
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_official_static_belief import environment_args
from scripts.evaluate_official_static_belief import (
    DEFAULT_SOURCE, BeliefInferenceDataset, load_rows, predict_beliefs,
)
from scripts.evaluate_online_rgb_closed_loop import select_from_current_rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/records.pt"))
    parser.add_argument("--belief-checkpoint", type=Path, default=Path(
        "runs/belief_transformer_stage1_relative/model_train3epoch.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--frozen-results", type=Path, default=Path(
        "runs/path_local_rgb_evidence/results.json"))
    parser.add_argument("--semantic-checkpoint", type=Path, default=Path(
        "runs/independent_semantic_trainseen_valseen_select/model.pt"))
    parser.add_argument("--type-features", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/"
        "runs/target_type_alignment/features.pt"))
    parser.add_argument("--cache", type=Path, default=Path(
        "runs/evidence_stop_head_belief/states.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/evidence_stop_head_belief"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument("--evidence-radius-m", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def selected_environment(split, records, protocol, batch_size):
    cropclient.load_image_cache()
    trajectories = load_mturk_trajectories(split, "all", protocol.altitude_m)
    selected = [trajectories[int(record["index"])] for record in records]
    episodes = generate_episodes_from_mturk_trajectories(
        get_city_refer_objects(), selected)
    if len(episodes) != len(records):
        raise ValueError(f"selected episode filtering mismatch: {len(episodes)} != {len(records)}")
    env = CityNavBatch.__new__(CityNavBatch)
    env.split = split
    env.args = environment_args(protocol, batch_size, 0)
    env.data = episodes
    env.batch_size = batch_size
    env.rank = 0; env.world_size = 1; env.ix = 0; env.nav_maps = []
    return env


def inference_plans(records, predicted_types, beliefs):
    return [{
        "modes": np.asarray(record["modes"], dtype=np.float64),
        "prior": np.asarray(record["prior"], dtype=np.float64),
        "observation_action_step": int(record["observation_action_step"]),
        "predicted_type": int(predicted_types[int(record["index"])]),
        "belief": np.asarray(belief, dtype=np.float32),
    } for record, belief in zip(records, beliefs)]


@torch.inference_mode()
def collect_batch_states(env, plans, semantic, args, device, beta,
                         protocol=OfficialProtocol()):
    observations = env._get_obs(random_direction=False)
    poses = [item["pose"] for item in observations]
    selected = [None] * len(plans)
    posterior = [plan["prior"].copy() for plan in plans]
    evidence = [np.zeros(2, dtype=np.float64) for _ in plans]
    ended = np.zeros(len(plans), dtype=bool)
    states = [[] for _ in plans]
    center_radius = protocol.success_distance_m / (
        2.0 * protocol.altitude_m) * args.view_size

    for outer_step in range(protocol.max_action_len):
        images = torch.from_numpy(np.stack([
            observation["rgb"] for observation in observations
        ])).permute(0, 3, 1, 2).float().to(device).div(255.0)
        probabilities = semantic(images).softmax(1).cpu().numpy()
        for index, observation in enumerate(observations):
            if ended[index]:
                continue
            plan = plans[index]
            if (selected[index] is None
                    and outer_step >= plan["observation_action_step"]):
                rank, value = select_from_current_rgb(
                    observation, plan, semantic, args, device, beta)
                selected[index] = rank; evidence[index] = value
                scores = np.clip(plan["prior"] + beta * value, 1e-8, None)
                posterior[index] = scores / scores.sum()
            rank = int(posterior[index].argmax())
            other = 1 - rank
            current = env.normalize_position(
                poses[index].xy, observation["map_name"], protocol.map_size_m)
            statistics = center_semantic_statistics(
                probabilities[index], plan["predicted_type"] + 1,
                center_radius)
            features = stop_features(
                current, plan["modes"][rank], plan["modes"][other],
                posterior[index], evidence[index], statistics,
                local_belief_statistics(plan["belief"], current),
                selected[index] is not None,
                protocol.max_action_len - outer_step,
                protocol.max_action_len, protocol.map_size_m)
            # Supervision only: target truth is never passed to stop_features.
            label = float(poses[index].xy.dist_to(
                env.batch[index].target_position.xy) <= protocol.success_distance_m)
            states[index].append({
                "features": features, "label": label,
                "eligible": selected[index] is not None,
                "outer_step": outer_step,
            })

            if selected[index] is not None:
                normalized = plan["modes"][selected[index]]
            else:
                normalized = plan["modes"].mean(0)
            destination = env.unnormalize_position(
                normalized, observation["map_name"], protocol.map_size_m)
            if (selected[index] is not None
                    and poses[index].xy.dist_to(destination) <= 5.0):
                ended[index] = True
                continue
            poses[index] = move_toward(
                poses[index], destination, protocol.move_iteration)

        if ended.all() or outer_step + 1 == protocol.max_action_len:
            break
        observations = env._get_obs(poses, random_direction=False)
    return states


def collect_split(split, records, predicted_types, beliefs, semantic, args,
                  device, beta):
    protocol = OfficialProtocol()
    plans = inference_plans(records, predicted_types, beliefs)
    env = selected_environment(split, records, protocol, args.batch_size)
    output, start = [], 0
    for _ in env:
        count = min(args.batch_size, len(plans) - start)
        indices = list(range(start, start + count))
        if count < args.batch_size:
            indices += list(range(args.batch_size - count))
        states = collect_batch_states(
            env, [plans[index] for index in indices], semantic, args, device, beta)
        output.extend(states[:count])
        start += count
        print(f"{split}: {start}/{len(plans)}", flush=True)
        if start >= len(plans):
            break
    return output


def load_or_collect(args, device):
    if args.cache.exists():
        saved = torch.load(args.cache, map_location="cpu", weights_only=False)
        first = saved["train_seen"][0][0]["features"]
        if len(first) != len(FEATURE_NAMES):
            raise ValueError("cached stop features use an incompatible schema")
        return saved
    records = torch.load(
        args.geometry_records, map_location="cpu", weights_only=False)
    train = records["train_seen"][:args.max_train or None]
    val = records["val_seen"][:args.max_val or None]
    frozen = json.loads(args.frozen_results.read_text())
    beta = float(frozen["selected_on_train_seen"]["rgb_weight"])
    type_features = torch.load(
        args.type_features, map_location="cpu", weights_only=True)
    type_model = train_probe(
        type_features["text"]["train_seen"],
        type_features["text"]["val_seen"], args.seed, 40, device)
    semantic = ProjectSemanticNet().to(device)
    semantic.load_state_dict(torch.load(
        args.semantic_checkpoint, map_location="cpu", weights_only=True))
    semantic.eval()
    predicted = {}
    beliefs = {}
    relation = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True)
    for split in ("train_seen", "val_seen"):
        predicted[split] = type_model(
            type_features["text"][split]["features"].float().to(device)
        ).argmax(-1).cpu().numpy()
        active_records = train if split == "train_seen" else val
        rows = load_rows(args.candidate_dir / f"{split}.jsonl")
        indices = [int(record["index"]) for record in active_records]
        selected_rows = [rows[index] for index in indices]
        dataset = BeliefInferenceDataset(
            selected_rows, relation[split][indices])
        belief_logits, _ = predict_beliefs(
            args.belief_checkpoint, dataset, device, 64)
        beliefs[split] = belief_logits.numpy()
    saved = {
        "train_seen": collect_split(
            "train_seen", train, predicted["train_seen"],
            beliefs["train_seen"], semantic, args, device, beta),
        "val_seen": collect_split(
            "val_seen", val, predicted["val_seen"],
            beliefs["val_seen"], semantic, args, device, beta),
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.cache)
    return saved


def flatten(episodes):
    features = torch.tensor(np.stack([
        state["features"] for episode in episodes for state in episode]))
    labels = torch.tensor([
        state["label"] for episode in episodes for state in episode],
        dtype=torch.float32)
    return features, labels


def episode_metrics(episodes, probabilities, threshold):
    cursor = 0; success = []; stop_steps = []
    for episode in episodes:
        count = len(episode)
        values = probabilities[cursor:cursor + count]
        eligible = [state["eligible"] for state in episode]
        chosen = first_stop_index(values, eligible, threshold)
        success.append(episode[chosen]["label"])
        stop_steps.append(episode[chosen]["outer_step"] + 1)
        cursor += count
    return {
        "sr": float(np.mean(success) * 100.0),
        "average_stop_step": float(np.mean(stop_steps)),
    }


def choose_threshold(episodes, probabilities):
    best = None
    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = episode_metrics(episodes, probabilities, float(threshold))
        key = metrics["sr"], -metrics["average_stop_step"], threshold
        if best is None or key > best[0]:
            best = key, float(threshold), metrics
    return best[1], best[2]


def train_head(saved, args, device):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_x, train_y = flatten(saved["train_seen"])
    val_x, _ = flatten(saved["val_seen"])
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-4)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    model = EvidenceStopHead().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    positive_weight = (train_y.numel() - train_y.sum()) / train_y.sum().clamp_min(1)
    best = None
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_x))
        model.train()
        for start in range(0, len(order), 512):
            index = order[start:start + 512]
            logits = model(train_x[index].to(device))
            loss = F.binary_cross_entropy_with_logits(
                logits, train_y[index].to(device), pos_weight=positive_weight.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            probability = model(val_x.to(device)).sigmoid().cpu().numpy()
        threshold, metrics = choose_threshold(saved["val_seen"], probability)
        key = metrics["sr"], -metrics["average_stop_step"], -epoch
        if best is None or key > best[0]:
            best = key, copy.deepcopy(model.state_dict()), epoch, threshold, metrics
    model.load_state_dict(best[1]); model.eval()
    return model, mean, std, best[2], best[3], best[4]


def probabilities(model, episodes, mean, std, device, ablate_center=False,
                  shuffle_center=False, ablate_belief=False,
                  shuffle_belief=False, seed=0):
    values, _ = flatten(episodes)
    if ablate_center:
        values[:, 6:9] = 0
    if shuffle_center:
        rng = np.random.default_rng(seed)
        order = torch.tensor(rng.permutation(len(values)))
        values[:, 6:9] = values[order, 6:9]
    if ablate_belief:
        values[:, 9:11] = 0
    if shuffle_belief:
        rng = np.random.default_rng(seed + 1)
        order = torch.tensor(rng.permutation(len(values)))
        values[:, 9:11] = values[order, 9:11]
    with torch.inference_mode():
        return model(((values - mean) / std).to(device)).sigmoid().cpu().numpy()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = load_or_collect(args, device)
    model, mean, std, epoch, threshold, selected = train_head(saved, args, device)
    val_probability = probabilities(
        model, saved["val_seen"], mean, std, device)
    zero_probability = probabilities(
        model, saved["val_seen"], mean, std, device, ablate_center=True)
    shuffled_probability = probabilities(
        model, saved["val_seen"], mean, std, device,
        shuffle_center=True, seed=args.seed)
    zero_belief_probability = probabilities(
        model, saved["val_seen"], mean, std, device, ablate_belief=True)
    shuffled_belief_probability = probabilities(
        model, saved["val_seen"], mean, std, device,
        shuffle_belief=True, seed=args.seed)
    baseline_probability = np.zeros_like(val_probability)
    result = {
        "experiment": "learned RGB-evidence stop head",
        "counts": {
            "train_episodes": len(saved["train_seen"]),
            "val_episodes": len(saved["val_seen"]),
            "train_states": sum(map(len, saved["train_seen"])),
            "val_states": sum(map(len, saved["val_seen"])),
        },
        "selected_on_val_seen": {
            "epoch": epoch, "threshold": threshold, **selected},
        "val_seen": {
            "deterministic_final_stop": episode_metrics(
                saved["val_seen"], baseline_probability, 1.0),
            "learned_stop": episode_metrics(
                saved["val_seen"], val_probability, threshold),
            "zero_center_semantics_control": episode_metrics(
                saved["val_seen"], zero_probability, threshold),
            "shuffled_center_semantics_control": episode_metrics(
                saved["val_seen"], shuffled_probability, threshold),
            "zero_local_belief_control": episode_metrics(
                saved["val_seen"], zero_belief_probability, threshold),
            "shuffled_local_belief_control": episode_metrics(
                saved["val_seen"], shuffled_belief_probability, threshold),
            "trajectory_oracle_sr": float(np.mean([
                any(state["label"] for state in episode)
                for episode in saved["val_seen"]]) * 100.0),
        },
        "feature_names": FEATURE_NAMES,
        "anti_leakage": {
            "target_use": "supervised within-20m label only",
            "target_in_feature_function": False,
            "training_split": "train_seen",
            "selection_split": "val_seen",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
        },
        "scope": "offline early-stop gate on exact official rollout states",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "mean": mean, "std": std,
        "threshold": threshold, "feature_names": FEATURE_NAMES,
    }, args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
