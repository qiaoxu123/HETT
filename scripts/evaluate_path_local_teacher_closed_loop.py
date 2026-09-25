#!/usr/bin/env python3
"""Official rollout ceiling using cached path-local point-cloud teacher scores."""

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
from multiagent.env import CityNavBatch
from scripts.evaluate_official_static_belief import (
    aligned_indices, environment_args, summarize_traces,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path(
        "runs/path_local_pointcloud_teacher/records.pt"))
    parser.add_argument("--teacher-results", type=Path, default=Path(
        "runs/path_local_pointcloud_teacher/results.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/path_local_pointcloud_teacher/official_closed_loop.json"))
    parser.add_argument("--evidence-source", choices=("pointcloud", "rgb"),
                        default="pointcloud")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--split", choices=("val_seen", "val_unseen"),
                        default="val_seen")
    return parser.parse_args()


def deployment_plans(records, beta):
    """Strip metrics-only fields before rollout and rank without targets."""
    plans = []
    for record in records:
        score = np.asarray(record["prior"]) + beta * np.asarray(record["evidence"])
        plans.append({
            "modes": np.asarray(record["modes"], dtype=np.float64),
            "selected_rank": int(score.argmax()),
            "observation_action_step": int(record["observation_action_step"]),
        })
    return plans


def rollout_teacher_plans(env, plans, protocol):
    """Progress toward the midpoint, observe once, then approach teacher rank."""
    observations = env._get_obs(random_direction=False)
    poses = [observation["pose"] for observation in observations]
    traces = [RolloutTrace([pose]) for pose in poses]
    ended = np.zeros(len(poses), dtype=bool)
    for outer_step in range(protocol.max_action_len):
        for index, observation in enumerate(observations):
            if ended[index]:
                continue
            trace, plan = traces[index], plans[index]
            trace.action_count += 1
            # observation_action_step counts movement actions. Until that pose
            # is reached, every action is still progress toward the mode pair.
            if outer_step < plan["observation_action_step"]:
                normalized = plan["modes"].mean(0)
                may_stop = False
            else:
                normalized = plan["modes"][plan["selected_rank"]]
                may_stop = True
            destination = env.unnormalize_position(
                normalized, observation["map_name"], protocol.map_size_m)
            if may_stop and poses[index].xy.dist_to(destination) <= 5.0:
                ended[index] = True
                trace.stop_step = outer_step + 1
                continue
            poses[index] = move_toward(
                poses[index], destination, protocol.move_iteration)
            trace.trajectory.append(poses[index])
            trace.movement_actions += 1
        if ended.all() or outer_step + 1 == protocol.max_action_len:
            break
        observations = env._get_obs(poses, random_direction=False)
        for index, is_ended in enumerate(ended):
            if not is_ended:
                traces[index].online_observations += 1
    return traces


def main():
    args = parse_args()
    protocol = OfficialProtocol(); protocol.validate()
    saved = torch.load(args.records, map_location="cpu", weights_only=False)
    records = saved[args.split] if isinstance(saved, dict) else saved
    if args.max_episodes:
        records = records[:args.max_episodes]
    teacher_result = json.loads(args.teacher_results.read_text())
    weight_key = ("pointcloud_weight" if args.evidence_source == "pointcloud"
                  else "rgb_weight")
    selection = teacher_result.get(
        "selected_on_train_seen", teacher_result.get("selected_before_val_unseen"))
    if selection is None:
        raise KeyError("result does not record a frozen weight source")
    beta = float(selection[weight_key])
    plans = deployment_plans(records, beta)
    env = CityNavBatch(
        args.split, environment_args(protocol, args.batch_size, args.max_episodes),
        batch_size=args.batch_size, seed=0, rank=0, world_size=1)
    results, all_traces = {}, []
    start = 0
    for _ in env:
        indices = aligned_indices(start, args.batch_size, len(plans))
        traces = rollout_teacher_plans(
            env, [plans[index] for index in indices], protocol)
        for item in attach_metric_only_ground_truth(env, traces):
            results[item["instr_id"]] = item
        valid_count = min(args.batch_size, len(plans) - start)
        all_traces.extend(traces[:valid_count])
        start += args.batch_size
        if start >= len(plans):
            break
    metrics, _ = env.eval_metrics(results)
    result = {
        "experiment": f"official path-local {args.evidence_source} closed loop",
        "split": args.split, "count": len(results),
        "official_metrics": metrics,
        "rollout_diagnostics": summarize_traces(
            all_traces, protocol.max_action_len),
        "teacher_weight_selected_on": "train_seen",
        "teacher_weight": beta,
        "privileged": args.evidence_source == "pointcloud",
        "anti_leakage": {
            "rollout_plan_fields": list(plans[0]),
            "target_in_rollout_plan": False,
            "target_in_rollout_function": False,
            "evidence_source": args.evidence_source,
            "pointcloud_used_at_runtime": args.evidence_source == "pointcloud",
            "val_unseen_used_for_selection": False,
            "test_unseen_loaded": False,
        },
        "scope": (
            "teacher ceiling under official actions" if args.evidence_source == "pointcloud"
            else "deployable RGB evidence under official actions; stop remains deterministic"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
