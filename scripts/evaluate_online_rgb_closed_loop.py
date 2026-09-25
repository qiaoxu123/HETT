#!/usr/bin/env python3
"""Strict online RGB inference inside the official CityNav rollout loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import (
    OfficialProtocol, RolloutTrace, attach_metric_only_ground_truth, move_toward,
)
from independent_nav.semantic import ProjectSemanticNet
from multiagent.env import CityNavBatch
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_official_static_belief import (
    aligned_indices, environment_args, summarize_traces,
)
from scripts.evaluate_path_local_pointcloud_teacher import normalized_to_world
from scripts.evaluate_path_local_rgb_evidence import (
    candidate_patches, semantic_evidence,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val_seen", "val_unseen"),
                        default="val_unseen")
    parser.add_argument("--geometry-records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/val_unseen_records.pt"))
    parser.add_argument("--frozen-results", type=Path, default=Path(
        "runs/path_local_rgb_evidence/val_unseen_results.json"))
    parser.add_argument("--semantic-checkpoint", type=Path, default=Path(
        "runs/independent_semantic_trainseen_valseen_select/model.pt"))
    parser.add_argument("--type-features", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/"
        "runs/target_type_alignment/features.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/path_local_rgb_evidence/val_unseen_strict_online.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument("--evidence-radius-m", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def inference_plans(records, predicted_types):
    """Only inference-visible state survives into the online rollout."""
    return [{
        "modes": np.asarray(record["modes"], dtype=np.float64),
        "prior": np.asarray(record["prior"], dtype=np.float64),
        "observation_action_step": int(record["observation_action_step"]),
        "predicted_type": int(predicted_types[int(record["index"])]),
        "cached_rank": int((np.asarray(record["prior"])
                            + 0.5 * np.asarray(record["evidence"])).argmax()),
    } for record in records]


@torch.inference_mode()
def select_from_current_rgb(observation, plan, model, args, device, beta):
    modes_world = np.stack([
        normalized_to_world(mode, observation["map_name"])
        for mode in plan["modes"]])
    patches = candidate_patches(
        observation["rgb"], observation["pose"], modes_world,
        args.view_size, args.patch_size, args.patch_side_m)
    images = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2)
    probability = model(images.float().to(device).div(255.0)).softmax(1).cpu().numpy()
    radius = args.evidence_radius_m / args.patch_side_m * args.patch_size
    target_class = plan["predicted_type"] + 1
    evidence = np.asarray([
        semantic_evidence(value, target_class, radius) for value in probability])
    return int((plan["prior"] + beta * evidence).argmax()), evidence


def rollout_online_rgb(env, plans, model, args, device, beta):
    """Acquire and consume current RGB at the decision step inside rollout."""
    observations = env._get_obs(random_direction=False)
    poses = [observation["pose"] for observation in observations]
    traces = [RolloutTrace([pose]) for pose in poses]
    selected = [None] * len(poses)
    evidence_rows = [None] * len(poses)
    ended = np.zeros(len(poses), dtype=bool)
    for outer_step in range(OfficialProtocol().max_action_len):
        for index, observation in enumerate(observations):
            if ended[index]:
                continue
            trace, plan = traces[index], plans[index]
            trace.action_count += 1
            if outer_step < plan["observation_action_step"]:
                normalized = plan["modes"].mean(0)
                may_stop = False
            else:
                if selected[index] is None:
                    selected[index], evidence_rows[index] = select_from_current_rgb(
                        observation, plan, model, args, device, beta)
                normalized = plan["modes"][selected[index]]
                may_stop = True
            destination = env.unnormalize_position(
                normalized, observation["map_name"], 410.0)
            if may_stop and poses[index].xy.dist_to(destination) <= 5.0:
                ended[index] = True
                trace.stop_step = outer_step + 1
                continue
            poses[index] = move_toward(poses[index], destination, 10)
            trace.trajectory.append(poses[index])
            trace.movement_actions += 1
        if ended.all() or outer_step + 1 == OfficialProtocol().max_action_len:
            break
        observations = env._get_obs(poses, random_direction=False)
        for index, is_ended in enumerate(ended):
            if not is_ended:
                traces[index].online_observations += 1
    return traces, selected, evidence_rows


def main():
    args = parse_args()
    protocol = OfficialProtocol(); protocol.validate()
    records = torch.load(
        args.geometry_records, map_location="cpu", weights_only=False)
    if isinstance(records, dict):
        records = records[args.split]
    if args.max_episodes:
        records = records[:args.max_episodes]
    frozen = json.loads(args.frozen_results.read_text())
    selection = frozen.get(
        "selected_before_val_unseen", frozen.get("selected_on_train_seen"))
    beta = float(selection["rgb_weight"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.load(args.type_features, map_location="cpu", weights_only=True)
    type_model = train_probe(
        features["text"]["train_seen"], features["text"]["val_seen"],
        args.seed, 40, device)
    with torch.inference_mode():
        predicted_type = type_model(
            features["text"][args.split]["features"].float().to(device)
        ).argmax(-1).cpu().numpy()
    plans = inference_plans(records, predicted_type)
    semantic = ProjectSemanticNet().to(device)
    semantic.load_state_dict(torch.load(
        args.semantic_checkpoint, map_location="cpu", weights_only=True))
    semantic.eval()

    env = CityNavBatch(
        args.split, environment_args(protocol, args.batch_size, args.max_episodes),
        batch_size=args.batch_size, seed=0, rank=0, world_size=1)
    results, all_traces, selected_all, cached_all = {}, [], [], []
    start = 0
    for _ in env:
        indices = aligned_indices(start, args.batch_size, len(plans))
        batch_plans = [plans[index] for index in indices]
        traces, selected, _ = rollout_online_rgb(
            env, batch_plans, semantic, args, device, beta)
        for item in attach_metric_only_ground_truth(env, traces):
            results[item["instr_id"]] = item
        valid_count = min(args.batch_size, len(plans) - start)
        all_traces.extend(traces[:valid_count])
        selected_all.extend(selected[:valid_count])
        cached_all.extend([
            plan["cached_rank"] for plan in batch_plans[:valid_count]])
        start += args.batch_size
        if start >= len(plans):
            break
    metrics, _ = env.eval_metrics(results)
    selected_array, cached_array = np.asarray(selected_all), np.asarray(cached_all)
    result = {
        "experiment": "strict online path-local RGB closed loop",
        "split": args.split, "count": len(results),
        "official_metrics": metrics,
        "rollout_diagnostics": summarize_traces(
            all_traces, protocol.max_action_len),
        "online_vs_cached_rank_match_percent": float(
            np.mean(selected_array == cached_array) * 100.0),
        "rgb_weight": beta,
        "anti_leakage": {
            "semantic_inference_location": "inside rollout on current obs[rgb]",
            "policy_inputs": (
                "current RGB, pose, predicted language type, belief modes and prior"),
            "target_in_plan_or_policy": False,
            "pointcloud_used_at_runtime": False,
            "val_unseen_used_for_selection": False,
            "test_unseen_loaded": False,
        },
        "scope": "strict online RGB result; deterministic predicted-goal stop remains",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

