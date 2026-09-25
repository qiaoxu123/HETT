#!/usr/bin/env python3
"""Strict 20-step upper bound for selecting among deterministic belief paths.

Every path is generated from the start pose and a target-free belief mode.  The
privileged oracle is used only after all K paths exist, to measure whether a
future RGB/language trajectory scorer could improve over the Top-1 path.
"""

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
        "runs/multitrajectory_oracle"))
    parser.add_argument("--modes", type=int, default=5)
    parser.add_argument("--minimum-separation-m", type=float, default=25.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--selection", type=Path,
        help="Optional frozen target-free scorer selections for protocol evaluation")
    return parser.parse_args()


def generate_mode_trajectory(start_pose, map_name, mode_xy, env, protocol):
    """Generate one deterministic official-action path without target access."""
    pose = start_pose
    trace = RolloutTrace([pose])
    destination = env.unnormalize_position(
        mode_xy, map_name, protocol.map_size_m)
    for outer_step in range(protocol.max_action_len):
        trace.action_count += 1
        if pose.xy.dist_to(destination) <= 5.0:
            trace.stop_step = outer_step + 1
            break
        pose = move_toward(pose, destination, protocol.move_iteration)
        trace.trajectory.append(pose)
        trace.movement_actions += 1
        trace.online_observations += 1
    return trace


def generate_trajectory_set(start_pose, map_name, modes, env, protocol):
    """Target-free K-path generator; the oracle is intentionally elsewhere."""
    return [generate_mode_trajectory(
        start_pose, map_name, mode, env, protocol) for mode in modes]


def privileged_select_trajectory(traces, target_xy):
    """Metric-only selector for the upper bound, never a deployable policy."""
    distances = [trace.trajectory[-1].xy.dist_to(target_xy) for trace in traces]
    return int(np.argmin(distances))


def main():
    args = parse_args()
    protocol = OfficialProtocol(); protocol.validate()
    rows = load_rows(args.candidate_dir / f"{args.split}.jsonl")
    relation = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True)[args.split]
    if args.max_episodes:
        rows, relation = rows[:args.max_episodes], relation[:args.max_episodes]
    dataset = BeliefInferenceDataset(rows, relation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beliefs, _ = predict_beliefs(args.checkpoint, dataset, device, 64)
    modes = [belief_distinct_topk_xy(
        belief, args.modes, args.minimum_separation_m) for belief in beliefs]
    targets = torch.tensor([row["target"] for row in rows])
    mode_tensor = torch.tensor(modes)
    distances = torch.linalg.vector_norm(
        mode_tensor - targets[:, None], dim=-1) * protocol.map_size_m
    learned_selection = None
    if args.selection:
        frozen = torch.load(args.selection, map_location="cpu", weights_only=True)
        learned_selection = frozen["selected"][:len(rows)].long()
        frozen_modes = frozen["modes"][:len(rows)]
        if len(learned_selection) != len(rows) or not torch.allclose(
                frozen_modes, mode_tensor, atol=1e-7):
            raise ValueError("frozen scorer selection does not align with belief modes")

    env = CityNavBatch(
        args.split, environment_args(protocol, args.batch_size, args.max_episodes),
        batch_size=args.batch_size, seed=0, rank=0, world_size=1)
    top1_results, oracle_results, learned_results = {}, {}, {}
    top1_traces, oracle_traces, learned_traces, oracle_ranks = [], [], [], []
    start = 0
    for _ in env:
        observations = env._get_obs(random_direction=False)
        indices = aligned_indices(start, args.batch_size, len(rows))
        generated = [generate_trajectory_set(
            observation["pose"], observation["map_name"], modes[index],
            env, protocol) for observation, index in zip(observations, indices)]
        chosen = [privileged_select_trajectory(
            paths, episode.target_position.xy)
            for paths, episode in zip(generated, env.batch)]
        selected_top1 = [paths[0] for paths in generated]
        selected_oracle = [paths[rank] for paths, rank in zip(generated, chosen)]
        selected_learned = ([paths[int(learned_selection[index])]
                             for paths, index in zip(generated, indices)]
                            if learned_selection is not None else [])
        for item in attach_metric_only_ground_truth(env, selected_top1):
            top1_results[item["instr_id"]] = item
        for item in attach_metric_only_ground_truth(env, selected_oracle):
            oracle_results[item["instr_id"]] = item
        if learned_selection is not None:
            for item in attach_metric_only_ground_truth(env, selected_learned):
                learned_results[item["instr_id"]] = item
        valid_count = min(args.batch_size, len(rows) - start)
        top1_traces.extend(selected_top1[:valid_count])
        oracle_traces.extend(selected_oracle[:valid_count])
        learned_traces.extend(selected_learned[:valid_count])
        oracle_ranks.extend(chosen[:valid_count])
        start += args.batch_size
        if start >= len(rows):
            break

    top1_metrics, _ = env.eval_metrics(top1_results)
    oracle_metrics, _ = env.eval_metrics(oracle_results)
    learned_metrics = (env.eval_metrics(learned_results)[0]
                       if learned_selection is not None else None)
    result = {
        "experiment": "strict deterministic Top-K trajectory selection upper bound",
        "split": args.split,
        "count": len(top1_results),
        "trajectory_hypotheses": args.modes,
        "minimum_mode_separation_m": args.minimum_separation_m,
        "endpoint_coverage": {
            "top1_hit20": float(distances[:, 0].le(20).float().mean() * 100),
            "topk_oracle_hit20": float(
                distances.amin(1).le(20).float().mean() * 100),
        },
        "A_top1_official": top1_metrics,
        "B_generated_paths": {
            "paths_per_episode": args.modes,
            "all_paths_target_free": True,
            "maximum_actions": protocol.max_action_len,
        },
        "C_current_belief_selector": {
            "selected_rank": 1,
            "official_metrics": top1_metrics,
            "meaning": "current belief score selects its Top-1 mode",
        },
        "D_privileged_trajectory_oracle": {
            "official_metrics": oracle_metrics,
            "selected_rank_histogram": {
                str(rank + 1): int(np.sum(np.asarray(oracle_ranks) == rank))
                for rank in range(args.modes)},
        },
        "E_frozen_target_free_scorer": ({
            "official_metrics": learned_metrics,
            "selection_file": str(args.selection.resolve()),
            "selected_rank_histogram": {
                str(rank + 1): int(learned_selection.eq(rank).sum())
                for rank in range(args.modes)},
        } if learned_selection is not None else None),
        "top1_rollout": summarize_traces(top1_traces, protocol.max_action_len),
        "oracle_rollout": summarize_traces(oracle_traces, protocol.max_action_len),
        "learned_rollout": (summarize_traces(
            learned_traces, protocol.max_action_len)
            if learned_selection is not None else None),
        "privileged": True,
        "anti_leakage": {
            "target_argument_in_path_generator": (
                "target" in inspect.signature(generate_trajectory_set).parameters),
            "target_use": "oracle path selection and final metrics only",
            "policy_claim": False,
            "test_unseen_loaded": False,
        },
        "decision_rule": (
            "Train diffusion only if D materially exceeds A; otherwise reject "
            "trajectory generation as the current bottleneck."),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.max_episodes}" if args.max_episodes else "_full"
    if learned_selection is not None:
        suffix += "_scored"
    output = args.output_dir / f"{args.split}_top{args.modes}{suffix}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); print("saved", output)


if __name__ == "__main__":
    main()
