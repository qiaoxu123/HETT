#!/usr/bin/env python3
"""Evaluate the unified eight-path rule library under the official protocol."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import (
    OfficialProtocol, RolloutTrace, attach_metric_only_ground_truth,
    belief_distinct_topk_xy, move_toward,
)
from independent_nav.trajectory_library import generate_rule_trajectory_set
from multiagent.env import CityNavBatch
from scripts.build_temporal_belief_dataset import pose_xy_yaw
from scripts.evaluate_official_static_belief import (
    DEFAULT_SOURCE, BeliefInferenceDataset, aligned_indices, environment_args,
    load_rows, predict_beliefs, summarize_traces,
)
from scripts.train_trajectory_hypothesis_scorer import normalized_start


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val_seen",), default="val_seen")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "runs/belief_transformer_stage1_relative/model_train3epoch.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/rule_trajectory_library"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=0)
    return parser.parse_args()


def rollout_encoded_trajectory(start_pose, map_name, hypothesis, env, protocol):
    """Execute one target-free encoded path using official movement actions."""
    pose = start_pose
    trace = RolloutTrace([pose])
    for step in range(hypothesis.movement_steps):
        trace.action_count += 1
        waypoint = hypothesis.positions[step + 1].tolist()
        destination = env.unnormalize_position(
            waypoint, map_name, protocol.map_size_m)
        pose = move_toward(pose, destination, protocol.move_iteration)
        trace.trajectory.append(pose)
        trace.movement_actions += 1
        trace.online_observations += 1
    if trace.action_count < protocol.max_action_len:
        trace.action_count += 1
        trace.stop_step = trace.action_count
    return trace


def trajectory_diversity(library):
    positions = torch.stack([item.positions for item in library])
    distances = torch.linalg.vector_norm(
        positions[:, None] - positions[None], dim=-1).mean(-1) * 410.0
    selected = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    return float(distances[selected].mean())


def main():
    args = parse_args(); protocol = OfficialProtocol(); protocol.validate()
    rows = load_rows(args.candidate_dir / f"{args.split}.jsonl")
    trajectories = json.loads((args.data_root / "processed_citynav" /
                               f"citynav_{args.split}.json").read_text())
    relation = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True)[args.split]
    if args.max_episodes:
        rows, trajectories = rows[:args.max_episodes], trajectories[:args.max_episodes]
        relation = relation[:args.max_episodes]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beliefs, _ = predict_beliefs(
        args.checkpoint, BeliefInferenceDataset(rows, relation), device, 64)
    modes = [belief_distinct_topk_xy(value, 5, 25.0) for value in beliefs]
    libraries = []
    for row, trajectory, hypotheses in zip(rows, trajectories, modes):
        start = normalized_start(trajectory["trajectory"][0], row["map_name"])
        _, _, yaw = pose_xy_yaw(trajectory["trajectory"][0])
        libraries.append(generate_rule_trajectory_set(start, yaw, hypotheses))

    targets = torch.tensor([row["target"] for row in rows])
    endpoints = torch.stack([
        torch.stack([item.positions[-1] for item in library])
        for library in libraries])
    endpoint_distance = torch.linalg.vector_norm(
        endpoints - targets[:, None], dim=-1) * protocol.map_size_m

    env = CityNavBatch(
        args.split, environment_args(protocol, args.batch_size, args.max_episodes),
        batch_size=args.batch_size, seed=0, rank=0, world_size=1)
    top1_results, oracle_results = {}, {}
    top1_traces, oracle_traces, selected_indices = [], [], []
    start_index = 0
    for _ in env:
        observations = env._get_obs(random_direction=False)
        indices = aligned_indices(start_index, args.batch_size, len(rows))
        generated = [[rollout_encoded_trajectory(
            observation["pose"], observation["map_name"], hypothesis,
            env, protocol) for hypothesis in libraries[index]]
            for observation, index in zip(observations, indices)]
        chosen = [int(np.argmin([
            trace.trajectory[-1].xy.dist_to(episode.target_position.xy)
            for trace in values]))
            for values, episode in zip(generated, env.batch)]
        selected_top1 = [values[0] for values in generated]
        selected_oracle = [values[index] for values, index in zip(generated, chosen)]
        for item in attach_metric_only_ground_truth(env, selected_top1):
            top1_results[item["instr_id"]] = item
        for item in attach_metric_only_ground_truth(env, selected_oracle):
            oracle_results[item["instr_id"]] = item
        valid_count = min(args.batch_size, len(rows) - start_index)
        top1_traces.extend(selected_top1[:valid_count])
        oracle_traces.extend(selected_oracle[:valid_count])
        selected_indices.extend(chosen[:valid_count])
        start_index += args.batch_size
        if start_index >= len(rows):
            break
    top1_metrics = env.eval_metrics(top1_results)[0]
    oracle_metrics = env.eval_metrics(oracle_results)[0]
    result = {
        "experiment": "eight target-free navigation-and-observation rule paths",
        "split": args.split, "count": len(rows),
        "endpoint_hit20": {
            "top1": float(endpoint_distance[:, 0].le(20).float().mean() * 100),
            "best_of_8": float(endpoint_distance.amin(1).le(20).float().mean() * 100),
        },
        "strict_top1": top1_metrics,
        "strict_oracle_best_of_8": oracle_metrics,
        "average_pairwise_path_distance_m": float(np.mean([
            trajectory_diversity(value) for value in libraries])),
        "budget": {
            "all_paths_reach_final_waypoint": all(
                item.reaches_final_waypoint for values in libraries for item in values),
            "maximum_movement_steps": max(
                item.movement_steps for values in libraries for item in values),
            "paths_above_20_steps": sum(
                item.movement_steps > 20 for values in libraries for item in values),
        },
        "oracle_selected_path_histogram": {
            libraries[0][index].name: int(np.sum(np.asarray(selected_indices) == index))
            for index in range(8)},
        "top1_rollout": summarize_traces(top1_traces, 20),
        "oracle_rollout": summarize_traces(oracle_traces, 20),
        "anti_leakage": {
            "target_in_library_generator": (
                "target" in inspect.signature(generate_rule_trajectory_set).parameters),
            "target_use": "oracle selection and final metrics only",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.max_episodes}" if args.max_episodes else "_full"
    output = args.output_dir / f"{args.split}{suffix}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); print("saved", output)


if __name__ == "__main__":
    main()
