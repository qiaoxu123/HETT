#!/usr/bin/env python3
"""Evaluate joint goal-progress and information actions in official rollout.

Unlike the earlier cached midpoint rule, every outer action is selected from
legal path-local waypoints using the current belief, pose and remaining
20-action budget.  RGB evidence is acquired once the two hypotheses are jointly
visible, after which the same action utility naturally commits to one mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.budget_aware_policy import choose_path_action
from independent_nav.official_protocol import (
    OfficialProtocol, RolloutTrace, attach_metric_only_ground_truth, move_toward,
)
from independent_nav.semantic import ProjectSemanticNet
from multiagent.env import CityNavBatch
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_official_static_belief import (
    aligned_indices, environment_args, summarize_traces,
)
from scripts.evaluate_online_rgb_closed_loop import select_from_current_rgb
from scripts.evaluate_path_local_pointcloud_teacher import (
    normalized_to_world, visible_in_official_crop,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train_seen", "val_seen"),
                        default="val_seen")
    parser.add_argument("--geometry-records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/records.pt"))
    parser.add_argument("--frozen-results", type=Path, default=Path(
        "runs/path_local_rgb_evidence/results.json"))
    parser.add_argument("--semantic-checkpoint", type=Path, default=Path(
        "runs/independent_semantic_trainseen_valseen_select/model.pt"))
    parser.add_argument("--type-features", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/"
        "runs/target_type_alignment/features.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/budget_aware_rgb_policy/val_seen.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--information-weight", type=float, default=0.5)
    parser.add_argument("--budget-weight", type=float, default=2.0)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument("--evidence-radius-m", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def inference_plans(records, predicted_types):
    """Strip all metric-only fields before policy execution."""
    return [{
        "modes": np.asarray(record["modes"], dtype=np.float64),
        "prior": np.asarray(record["prior"], dtype=np.float64),
        "predicted_type": int(predicted_types[int(record["index"])]),
    } for record in records]


def jointly_visible(observation, modes):
    world = [normalized_to_world(mode, observation["map_name"]) for mode in modes]
    return all(visible_in_official_crop(observation["pose"], point) for point in world)


def rollout_budget_aware(env, plans, model, args, device, beta,
                         protocol=OfficialProtocol()):
    """Online rollout whose signature deliberately excludes target truth."""
    observations = env._get_obs(random_direction=False)
    poses = [observation["pose"] for observation in observations]
    traces = [RolloutTrace([pose]) for pose in poses]
    posterior = [plan["prior"].copy() for plan in plans]
    evidence_used = np.zeros(len(plans), dtype=bool)
    observation_steps = [None] * len(plans)
    action_names = [[] for _ in plans]
    ended = np.zeros(len(plans), dtype=bool)

    for outer_step in range(protocol.max_action_len):
        for index, observation in enumerate(observations):
            if ended[index]:
                continue
            trace, plan = traces[index], plans[index]
            trace.action_count += 1
            if (not evidence_used[index]
                    and jointly_visible(observation, plan["modes"])):
                _, evidence = select_from_current_rgb(
                    observation, plan, model, args, device, beta)
                scores = np.clip(plan["prior"] + beta * evidence, 1e-8, None)
                posterior[index] = scores / scores.sum()
                evidence_used[index] = True
                observation_steps[index] = outer_step

            committed = int(posterior[index].argmax())
            destination = env.unnormalize_position(
                plan["modes"][committed], observation["map_name"],
                protocol.map_size_m)
            if (evidence_used[index]
                    and poses[index].xy.dist_to(destination) <= 5.0):
                ended[index] = True
                trace.stop_step = outer_step + 1
                action_names[index].append("stop")
                continue

            current = env.normalize_position(
                poses[index].xy, observation["map_name"], protocol.map_size_m)
            remaining = protocol.max_action_len - outer_step
            action_probabilities = posterior[index]
            if evidence_used[index]:
                # Once the single planned observation has been consumed there
                # is no additional information source in this gate.  Continuing
                # to optimize expected distance to both modes converges to their
                # midpoint and wastes the action budget.  Commit to the current
                # posterior winner without consulting ground truth.
                action_probabilities = np.zeros_like(posterior[index])
                action_probabilities[committed] = 1.0
            action = choose_path_action(
                current, plan["modes"], action_probabilities, remaining, protocol,
                information_weight=args.information_weight,
                budget_weight=args.budget_weight)
            waypoint = env.unnormalize_position(
                action.xy, observation["map_name"], protocol.map_size_m)
            poses[index] = move_toward(
                poses[index], waypoint, protocol.move_iteration)
            trace.trajectory.append(poses[index])
            trace.movement_actions += 1
            action_names[index].append(action.name)

        if ended.all() or outer_step + 1 == protocol.max_action_len:
            break
        observations = env._get_obs(poses, random_direction=False)
        for index, is_ended in enumerate(ended):
            if not is_ended:
                traces[index].online_observations += 1
    return traces, posterior, evidence_used, observation_steps, action_names


def main():
    args = parse_args()
    protocol = OfficialProtocol(); protocol.validate()
    saved = torch.load(args.geometry_records, map_location="cpu", weights_only=False)
    records = saved[args.split] if isinstance(saved, dict) else saved
    if args.max_episodes:
        records = records[:args.max_episodes]
    frozen = json.loads(args.frozen_results.read_text())
    selection = frozen.get("selected_on_train_seen")
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
    results, all_traces, used_all, steps_all, names_all = {}, [], [], [], []
    start = 0
    for _ in env:
        indices = aligned_indices(start, args.batch_size, len(plans))
        batch_plans = [plans[index] for index in indices]
        traces, _, used, steps, names = rollout_budget_aware(
            env, batch_plans, semantic, args, device, beta, protocol)
        for item in attach_metric_only_ground_truth(env, traces):
            results[item["instr_id"]] = item
        count = min(args.batch_size, len(plans) - start)
        all_traces.extend(traces[:count]); used_all.extend(used[:count])
        steps_all.extend(steps[:count]); names_all.extend(names[:count])
        start += args.batch_size
        if start >= len(plans):
            break

    metrics, _ = env.eval_metrics(results)
    counts = Counter(name for values in names_all for name in values)
    valid_steps = [value for value in steps_all if value is not None]
    result = {
        "experiment": "budget-aware joint progress-information RGB policy",
        "split": args.split, "count": len(results),
        "official_metrics": metrics,
        "rollout_diagnostics": summarize_traces(
            all_traces, protocol.max_action_len),
        "policy_diagnostics": {
            "rgb_evidence_acquired_percent": float(np.mean(used_all) * 100.0),
            "mean_evidence_outer_step": (
                float(np.mean(valid_steps)) if valid_steps else None),
            "action_counts": dict(sorted(counts.items())),
        },
        "weights": {
            "information": args.information_weight,
            "budget": args.budget_weight, "rgb": beta,
        },
        "anti_leakage": {
            "policy_inputs": (
                "current RGB/pose, belief modes/prior, predicted language type, "
                "remaining action budget"),
            "target_in_plan_or_rollout": False,
            "pointcloud_at_runtime": False,
            "test_unseen_loaded": False,
        },
        "scope": (
            "rule-based action-utility gate at fixed 50 m; deterministic "
            "predicted-goal stop remains and no learned action head is claimed"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
