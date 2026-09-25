#!/usr/bin/env python3
"""Upper bound for one path-local observation under the official 20-step budget.

This is explicitly privileged: after the first target-independent path action,
an oracle uses the reference target to choose between two distinct belief modes.
It measures whether a perfect future RGB/point-cloud teacher could justify the
action design. It is not a deployable navigation result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import (
    OfficialProtocol, RolloutTrace, attach_metric_only_ground_truth,
    belief_distinct_topk_xy, generate_budgeted_waypoints, move_toward,
)
from multiagent.env import CityNavBatch
from scripts.evaluate_official_static_belief import (
    DEFAULT_SOURCE, BeliefInferenceDataset, aligned_indices, environment_args,
    load_rows, predict_beliefs, summarize_traces,
)


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
        "runs/budgeted_action_oracle"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    return parser.parse_args()


def choose_path_waypoint(current, top1, top2, protocol):
    """Target-independent first action: prefer the candidate midpoint path."""
    candidates = generate_budgeted_waypoints(
        current, top1, top2, protocol.max_action_len, protocol)
    by_name = {candidate.name: candidate.xy for candidate in candidates}
    return by_name.get("toward_midpoint", by_name.get("toward_top1", top1))


def normalized_position(env, pose, map_name, protocol):
    return env.normalize_position(pose.xy, map_name, protocol.map_size_m)


def rollout_perfect_two_mode_observation(env, modes, protocol):
    observations = env._get_obs(random_direction=False)
    poses = [item["pose"] for item in observations]
    traces = [RolloutTrace([pose]) for pose in poses]

    # Action 1 is selected without access to the target.
    for index, observation in enumerate(observations):
        current = normalized_position(
            env, poses[index], observation["map_name"], protocol)
        waypoint = choose_path_waypoint(
            current, modes[index][0], modes[index][1], protocol)
        destination = env.unnormalize_position(
            waypoint, observation["map_name"], protocol.map_size_m)
        poses[index] = move_toward(
            poses[index], destination, protocol.move_iteration)
        traces[index].trajectory.append(poses[index])
        traces[index].action_count = 1
        traces[index].movement_actions = 1

    observations = env._get_obs(poses, random_direction=False)
    for trace in traces:
        trace.online_observations += 1

    # Privileged teacher label only: emulate a perfect disambiguating view.
    selected = []
    for index, episode in enumerate(env.batch):
        target = np.asarray(env.normalize_position(
            episode.target_position.xy, episode.map_name, protocol.map_size_m))
        selected.append(min(
            modes[index], key=lambda mode: np.linalg.norm(np.asarray(mode) - target)))

    ended = np.zeros(len(poses), dtype=bool)
    for outer_step in range(1, protocol.max_action_len):
        for index, observation in enumerate(observations):
            if ended[index]:
                continue
            trace = traces[index]
            trace.action_count += 1
            destination = env.unnormalize_position(
                selected[index], observation["map_name"], protocol.map_size_m)
            if poses[index].xy.dist_to(destination) <= 5.0:
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
    return traces, selected


def main():
    args = parse_args()
    protocol = OfficialProtocol(); protocol.validate()
    row_path = args.candidate_dir / f"{args.split}.jsonl"
    rows = load_rows(row_path)
    relation = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True)[args.split]
    if args.max_episodes:
        rows, relation = rows[:args.max_episodes], relation[:args.max_episodes]
    dataset = BeliefInferenceDataset(rows, relation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beliefs, _ = predict_beliefs(args.checkpoint, dataset, device, 64)
    modes = [belief_distinct_topk_xy(item, 2) for item in beliefs]

    targets = torch.tensor([row["target"] for row in rows])
    mode_tensor = torch.tensor(modes)
    distances = torch.linalg.vector_norm(
        mode_tensor - targets[:, None], dim=-1) * protocol.map_size_m

    env = CityNavBatch(
        args.split, environment_args(protocol, args.batch_size, args.max_episodes),
        batch_size=args.batch_size, seed=0, rank=0, world_size=1)
    results, all_traces = {}, []
    start = 0
    for _ in env:
        indices = aligned_indices(start, args.batch_size, len(rows))
        traces, _ = rollout_perfect_two_mode_observation(
            env, [modes[index] for index in indices], protocol)
        for item in attach_metric_only_ground_truth(env, traces):
            results[item["instr_id"]] = item
        valid_count = min(args.batch_size, len(rows) - start)
        all_traces.extend(traces[:valid_count])
        start += args.batch_size
        if start >= len(rows):
            break

    metrics, _ = env.eval_metrics(results)
    result = {
        "experiment": "privileged two-mode path-local action upper bound",
        "split": args.split, "count": len(results),
        "static_top1_hit20": float(distances[:, 0].le(20).float().mean() * 100),
        "two_mode_oracle_hit20": float(
            distances.amin(1).le(20).float().mean() * 100),
        "official_metrics": metrics,
        "rollout_diagnostics": summarize_traces(
            all_traces, protocol.max_action_len),
        "privileged": True,
        "meaning": (
            "Action 1 is target-independent and path-local; after that action, "
            "ground truth chooses between two spatially distinct belief modes. "
            "This is an upper bound for a future observation teacher, not a model result."),
        "test_unseen_loaded": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.max_episodes}" if args.max_episodes else "_full"
    output = args.output_dir / f"{args.split}{suffix}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); print("saved", output)


if __name__ == "__main__":
    main()

