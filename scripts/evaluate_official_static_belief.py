#!/usr/bin/env python3
"""Evaluate the unified static belief model under the frozen HETT protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.belief_transformer import BeliefTransformer, rasterize_landmarks
from independent_nav.official_protocol import (
    OfficialProtocol, attach_metric_only_ground_truth, belief_argmax_xy,
    canonical_instruction, rollout_static_goals,
)
from multiagent.env import CityNavBatch
from scripts.train_relation_geometry_ranker import structured_relation_cues


DEFAULT_SOURCE = Path(
    "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/runs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val_seen", "val_unseen"),
                        default="val_seen")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "runs/belief_transformer_stage1_relative/model_train3epoch.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/official_static_belief"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class BeliefInferenceDataset(Dataset):
    """Inference inputs only; target coordinates are never materialized."""

    def __init__(self, rows, relation_features, grid_size=20, max_landmarks=9):
        if relation_features.shape[:2] != (len(rows), max_landmarks):
            raise ValueError("relation features do not align with candidate rows")
        self.relation = relation_features
        self.cues = structured_relation_cues(rows, max_landmarks)
        self.centers = torch.zeros(len(rows), max_landmarks, 2)
        self.boundaries = torch.zeros(len(rows), max_landmarks, 16, 2)
        self.valid = torch.zeros(len(rows), max_landmarks, dtype=torch.bool)
        for row_index, row in enumerate(rows):
            count = min(len(row.get("landmark_centers", [])),
                        len(row.get("landmark_boundaries", [])), max_landmarks)
            if count:
                self.centers[row_index, :count] = torch.tensor(
                    row["landmark_centers"][:count])
                self.boundaries[row_index, :count] = torch.tensor(
                    row["landmark_boundaries"][:count])
                self.valid[row_index, :count] = True
        self.masks = rasterize_landmarks(
            self.boundaries, self.valid, grid_size).cpu()
        union = self.masks.any(1).float()
        pooled = torch.nn.functional.max_pool2d(union[:, None], 3, 1, 1)[:, 0]
        self.map_bev = torch.stack((union, (pooled - union).clamp_min(0)), 1)

    def __len__(self):
        return len(self.relation)

    def __getitem__(self, index):
        return (self.relation[index], self.cues[index], self.centers[index],
                self.boundaries[index], self.valid[index], self.masks[index],
                self.map_bev[index])


@torch.inference_mode()
def predict_goals(checkpoint_path, dataset, device, batch_size):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = BeliefTransformer(
        grid_size=int(config["grid_size"]), hidden=int(config["hidden"]),
        layers=int(config["layers"]))
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    goals = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits = model(*(value.to(device) for value in batch)).cpu()
        goals.extend(belief_argmax_xy(item) for item in logits)
    return goals, config


def environment_args(protocol, batch_size, max_episodes):
    return SimpleNamespace(
        altitude=protocol.altitude_m, batch_size=batch_size,
        grid_size=protocol.grid_size, map_meters=protocol.map_size_m,
        map_size=240, map_shape=(240, 240),
        map_pixels_per_meter=240 / protocol.map_size_m,
        max_episodes=max_episodes, success_dist=protocol.success_distance_m,
    )


def aligned_indices(start, count, total):
    primary = list(range(start, min(start + count, total)))
    return primary + list(range(count - len(primary)))


def summarize_traces(traces, max_actions):
    actions = np.asarray([trace.action_count for trace in traces])
    stopped = [trace.stop_step for trace in traces if trace.stop_step is not None]
    return {
        "average_actions": float(actions.mean()),
        "average_movement_actions": float(np.mean(
            [trace.movement_actions for trace in traces])),
        "average_online_observations": float(np.mean(
            [trace.online_observations for trace in traces])),
        "explicit_stop_rate": 100.0 * len(stopped) / len(traces),
        "average_stop_step": float(np.mean(stopped)) if stopped else None,
        "episodes_over_20_actions": int(np.count_nonzero(actions > max_actions)),
        "maximum_actions": int(actions.max()),
    }


def main():
    args = parse_args()
    protocol = OfficialProtocol()
    protocol.validate()
    row_path = args.candidate_dir / f"{args.split}.jsonl"
    rows = load_rows(row_path)
    relation = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True)[args.split]
    if args.max_episodes:
        rows = rows[:args.max_episodes]
        relation = relation[:args.max_episodes]
    inference_data = BeliefInferenceDataset(rows, relation)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    goals, model_config = predict_goals(
        args.checkpoint, inference_data, device, max(1, args.batch_size * 16))

    env = CityNavBatch(
        args.split, environment_args(protocol, args.batch_size, args.max_episodes),
        batch_size=args.batch_size, seed=0, rank=0, world_size=1)
    results, traces_all = {}, []
    start = 0
    for _ in env:
        indices = aligned_indices(start, args.batch_size, len(rows))
        for index, episode in zip(indices, env.batch):
            row = rows[index]
            if (row["map_name"] != episode.map_name
                    or canonical_instruction(row["description"])
                    != canonical_instruction(episode.target_description)):
                raise RuntimeError(f"candidate/environment mismatch at {index}")
        traces = rollout_static_goals(
            env, [goals[index] for index in indices], protocol)
        # Ground truth enters only here, after every action has been selected.
        for item in attach_metric_only_ground_truth(env, traces):
            results[item["instr_id"]] = item
        traces_all.extend(traces[:min(args.batch_size, len(rows) - start)])
        start += args.batch_size
        if start >= len(rows):
            break

    metrics, _ = env.eval_metrics(results)
    result = {
        "experiment": "C: unified static belief + official online rollout",
        "split": args.split, "count": len(results),
        "official_metrics": metrics,
        "rollout_diagnostics": summarize_traces(
            traces_all, protocol.max_action_len),
        "protocol": protocol.__dict__,
        "policy": {
            "belief": "language + independent named landmarks + map BEV",
            "goal": "argmax center of the 20x20 predicted belief",
            "motion": "exact HETT stage-1 discrete lookahead, 10 iterations",
            "stop": "within 5 m of the model-predicted goal; no target access",
            "rgb": "fresh online RGB acquired each step but intentionally unused in C",
            "active_view": False, "dynamic_height": False,
        },
        "anti_leakage": {
            "target_in_inference_dataset": False,
            "target_in_policy_or_rollout_arguments": False,
            "ground_truth_attached_after_action_generation": True,
            "candidate_environment_alignment": (
                "map + case-folded alphanumeric instruction identity"),
            "test_unseen_loaded": False,
        },
        "artifacts": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "candidate_rows": str(row_path.resolve()),
            "candidate_rows_sha256": sha256(row_path),
            "relation_cache": str(args.relation_cache.resolve()),
            "relation_cache_sha256": sha256(args.relation_cache),
            "model_config": model_config,
        },
        "scope": (
            "strict stage-C static closed-loop test; no claim about RGB, active view, "
            "learned stopping, or dynamic altitude"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.max_episodes}" if args.max_episodes else "_full"
    output = args.output_dir / f"{args.split}{suffix}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
