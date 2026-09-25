#!/usr/bin/env python3
"""Official val_seen rollout for frozen path-local verifier choices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import (
    OfficialProtocol, attach_metric_only_ground_truth,
)
from independent_nav.path_local_verifier import PathLocalCandidateVerifier
from multiagent.env import CityNavBatch
from scripts.evaluate_official_static_belief import (
    aligned_indices, environment_args, summarize_traces,
)
from scripts.evaluate_path_local_teacher_closed_loop import rollout_teacher_plans
from scripts.train_path_local_candidate_verifier import predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "runs/path_local_candidate_verifier/train8192/model.pt"))
    parser.add_argument("--features", type=Path, default=Path(
        "runs/path_local_candidate_verifier/features_8192.pt"))
    parser.add_argument("--records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/expanded_train_records.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/path_local_candidate_verifier/train8192/official_val_seen.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    return parser.parse_args()


def deployment_plans(records, selected):
    return [{
        "modes": record["modes"],
        "selected_rank": int(rank),
        "observation_action_step": int(record["observation_action_step"]),
    } for record, rank in zip(records, selected)]


def main():
    args = parse_args(); protocol = OfficialProtocol(); protocol.validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.features, map_location="cpu", weights_only=False)
    val = saved["val_seen"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PathLocalCandidateVerifier(int(checkpoint["vision_dim"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    selected = predict(model, val, device).argmax(-1).tolist()
    records = torch.load(
        args.records, map_location="cpu", weights_only=False)["val_seen"]
    if args.max_episodes:
        records = records[:args.max_episodes]; selected = selected[:args.max_episodes]
    plans = deployment_plans(records, selected)

    env = CityNavBatch(
        "val_seen", environment_args(protocol, args.batch_size, args.max_episodes),
        batch_size=args.batch_size, seed=0, rank=0, world_size=1)
    results, traces_all, start = {}, [], 0
    for _ in env:
        indices = aligned_indices(start, args.batch_size, len(plans))
        traces = rollout_teacher_plans(
            env, [plans[index] for index in indices], protocol)
        for item in attach_metric_only_ground_truth(env, traces):
            results[item["instr_id"]] = item
        count = min(args.batch_size, len(plans) - start)
        traces_all.extend(traces[:count]); start += args.batch_size
        if start >= len(plans):
            break
    metrics, _ = env.eval_metrics(results)
    result = {
        "experiment": "frozen path-local instance verifier official rollout",
        "split": "val_seen", "count": len(results),
        "official_metrics": metrics,
        "rollout_diagnostics": summarize_traces(
            traces_all, protocol.max_action_len),
        "anti_leakage": {
            "plan_fields": list(plans[0]), "target_in_plan": False,
            "target_in_rollout": False, "pointcloud_at_runtime": False,
            "val_unseen_loaded": False, "test_unseen_loaded": False,
        },
        "scope": (
            "cached frozen verifier choice with exact official online motion; "
            "strict in-rollout SigLIP execution follows only if positive"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
