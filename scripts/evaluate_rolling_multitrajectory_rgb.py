#!/usr/bin/env python3
"""Strict 20-action rolling evaluation of eight anchored trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import (
    OfficialProtocol, RolloutTrace, attach_metric_only_ground_truth, move_toward,
)
from independent_nav.semantic import ProjectSemanticNet
from independent_nav.trajectory_diffusion import (
    ConditionalUnet1D, bounded_endpoint_projection, ddim_sample,
)
from independent_nav.trajectory_library import (
    TrajectoryHypothesis, wrap_angle,
)
from independent_nav.trajectory_scorer import MultiObjectiveTrajectoryScorer
from multiagent.env import CityNavBatch
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_official_static_belief import (
    DEFAULT_SOURCE, BeliefInferenceDataset, environment_args, load_rows,
    predict_beliefs, summarize_traces,
)
from scripts.evaluate_online_rgb_closed_loop import select_from_current_rgb
from scripts.evaluate_path_local_pointcloud_teacher import normalized_to_world
from scripts.train_anchored_trajectory_scorer import SLOT_TO_MODE
from scripts.train_relation_geometry_ranker import build_split
from scripts.train_trajectory_hypothesis_scorer import belief_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val_seen",), default="val_seen")
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--belief-checkpoint", type=Path, default=Path(
        "runs/belief_transformer_stage1_relative/model_train3epoch.pt"))
    parser.add_argument("--scorer-checkpoint", type=Path, default=Path(
        "runs/anchored_trajectory_scorer_multiobjective_train1024_holdout256/model.pt"))
    parser.add_argument("--diffusion-checkpoint", type=Path, default=Path(
        "runs/trajectory_diffusion_slot_train1024_holdout256/model.pt"))
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--semantic-checkpoint", type=Path, default=Path(
        "runs/independent_semantic_trainseen_valseen_select/model.pt"))
    parser.add_argument("--type-features", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/"
        "runs/target_type_alignment/features.pt"))
    parser.add_argument("--rgb-results", type=Path, default=Path(
        "runs/path_local_rgb_evidence/results.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/rolling_multitrajectory_rgb/val_seen_32.json"))
    parser.add_argument("--max-episodes", type=int, default=32)
    parser.add_argument("--execute-steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument("--evidence-radius-m", type=float, default=10.0)
    parser.add_argument("--switch-margin", type=float, default=0.0,
                        help="Optional route hysteresis; zero keeps RGB corrections unrestricted")
    parser.add_argument("--commit-stable-replans", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def route_features(library, belief_map):
    action = torch.stack([value.actions for value in library])
    valid = torch.stack([value.valid for value in library])
    position = torch.stack([value.positions[1:] for value in library])
    standardized = (belief_map - belief_map.mean()) / belief_map.std().clamp_min(1e-5)
    grid = standardized.shape[-1]
    column = (position[..., 0] * grid).long().clamp(0, grid - 1)
    row = (position[..., 1] * grid).long().clamp(0, grid - 1)
    sampled = standardized[row, column][..., None]
    return torch.cat((
        action, position, sampled, valid[..., None].float()), -1)


def prepare(args, device):
    rows = load_rows(args.candidate_dir / f"{args.split}.jsonl")
    relation = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True)[args.split]
    if args.max_episodes:
        rows, relation = rows[:args.max_episodes], relation[:args.max_episodes]
    inference = BeliefInferenceDataset(rows, relation)
    beliefs, _ = predict_beliefs(
        args.belief_checkpoint, inference, device, 64)
    from independent_nav.official_protocol import belief_distinct_topk_xy
    modes = [belief_distinct_topk_xy(value, 5, 25.0) for value in beliefs]
    endpoints, enriched = [], []
    for row, hypotheses in zip(rows, modes):
        anchors = [hypotheses[index] for index in SLOT_TO_MODE]
        endpoints.append(anchors)
        value = copy.copy(row); value["candidates"] = [list(x) for x in anchors]
        enriched.append(value)
    static = build_split(enriched, relation, 9)
    static["belief_feature"] = belief_features(beliefs, endpoints)
    for slot, mode in enumerate(SLOT_TO_MODE):
        static["belief_feature"][:, slot, 3] = -mode / 4.0
    return rows, inference, beliefs, modes, static


@torch.inference_mode()
def score_routes(model, static, index, route, device):
    return model(
        static["relation"][index:index + 1].to(device),
        static["cue"][index:index + 1].to(device),
        static["geometry"][index:index + 1].to(device),
        static["valid"][index:index + 1].to(device),
        route[None].to(device),
        static["belief_feature"][index:index + 1].to(device))[0].cpu().numpy()


@torch.inference_mode()
def sample_diffusion_library(model, inference, static, index, belief, modes,
                             start, yaw, remaining, args, device, protocol):
    standardized = (belief - belief.mean()) / belief.std().clamp_min(1e-5)
    condition = (
        standardized[None], static["relation"][index:index + 1],
        static["cue"][index:index + 1], inference.centers[index:index + 1],
        inference.valid[index:index + 1],
        torch.tensor([[start[0], start[1], math.sin(yaw), math.cos(yaw),
                       remaining / protocol.max_action_len]]),
    )
    sampled, length = ddim_sample(
        model, condition, samples=8, diffusion_steps=args.diffusion_steps,
        inference_steps=args.inference_steps)
    endpoints = torch.tensor([[modes[slot] for slot in SLOT_TO_MODE]])
    step_scale = protocol.maximum_translation_per_action_m / protocol.map_size_m
    required = torch.ceil(torch.linalg.vector_norm(
        endpoints - torch.tensor(start)[None, None], dim=-1) / step_scale).long()
    # Rolling replanning must not renew an unrestricted detour budget at every
    # cycle. Allow at most two sensing steps beyond the optimistic shortest path.
    maximum = torch.minimum(required + 2, torch.full_like(required, remaining))
    length = torch.minimum(length, maximum.to(length.device))
    positions, effective = bounded_endpoint_projection(
        sampled, length, torch.tensor([start]), endpoints,
        step_scale)
    output = []
    for slot in range(8):
        count = min(int(effective[0, slot]), remaining)
        action = torch.zeros(20, 4); valid = torch.zeros(20, dtype=torch.bool)
        heading = float(yaw)
        for step in range(count):
            delta = positions[0, slot, step + 1] - positions[0, slot, step]
            next_heading = math.atan2(-float(delta[1]), float(delta[0]))
            delta_yaw = wrap_angle(next_heading - heading)
            action[step] = torch.tensor((
                float(delta[0] / step_scale), float(delta[1] / step_scale),
                math.sin(delta_yaw), math.cos(delta_yaw)))
            valid[step] = True; heading = next_heading
        output.append(TrajectoryHypothesis(
            name=f"diffusion_slot_{slot}", actions=action, valid=valid,
            positions=positions[0, slot], movement_steps=count,
            path_length_m=float(torch.linalg.vector_norm(
                positions[0, slot, 1:count + 1]
                - positions[0, slot, :count], dim=-1).sum() * protocol.map_size_m),
            reaches_final_waypoint=(count == int(effective[0, slot]))))
    return output


def rollout_one(env, observation, index, modes, belief, inference, static,
                diffusion, scorer,
                semantic, predicted_type, beta, args, device, protocol):
    pose = observation["pose"]
    trace = RolloutTrace([pose])
    evidence_sum = np.zeros(8, dtype=np.float64)
    evidence_count = 0
    selected_history, replans, committed_actions = [], 0, 0
    previous_selected = None
    while trace.action_count < protocol.max_action_len:
        current = env.normalize_position(
            pose.xy, observation["map_name"], protocol.map_size_m)
        yaw = float(pose.yaw)
        remaining = protocol.max_action_len - trace.action_count
        library = sample_diffusion_library(
            diffusion, inference, static, index, belief, modes, current, yaw,
            remaining, args, device, protocol)
        route = route_features(library, belief)
        score = score_routes(scorer, static, index, route, device)
        if evidence_count:
            score = score + beta * (evidence_sum / evidence_count)
        selected = int(score.argmax())
        if previous_selected is not None and args.switch_margin > 0:
            previous_mode = modes[SLOT_TO_MODE[previous_selected]]
            required_previous = int(math.ceil(
                np.linalg.norm(np.asarray(previous_mode) - np.asarray(current))
                * protocol.map_size_m / protocol.maximum_translation_per_action_m))
            advantage = float(score[selected] - score[previous_selected])
            if (advantage < args.switch_margin
                    or remaining <= required_previous + 2):
                selected = previous_selected
        previous_selected = selected
        selected_history.append(selected); replans += 1
        selected_modes = [SLOT_TO_MODE[value] for value in selected_history]
        stable = (evidence_count > 0
                  and len(selected_modes) >= args.commit_stable_replans
                  and len(set(selected_modes[-args.commit_stable_replans:])) == 1)
        endpoint = library[selected].positions[-1].tolist()
        destination = env.unnormalize_position(
            endpoint, observation["map_name"], protocol.map_size_m)
        if evidence_count and pose.xy.dist_to(destination) <= 5.0:
            trace.action_count += 1; trace.stop_step = trace.action_count
            break
        movement = min(args.execute_steps, library[selected].movement_steps,
                       protocol.max_action_len - trace.action_count)
        if movement <= 0:
            trace.action_count += 1; trace.stop_step = trace.action_count
            break
        for step in range(movement):
            if stable:
                waypoint = destination; committed_actions += 1
            else:
                waypoint = env.unnormalize_position(
                    library[selected].positions[step + 1].tolist(),
                    observation["map_name"], protocol.map_size_m)
            pose = move_toward(pose, waypoint, protocol.move_iteration)
            trace.trajectory.append(pose); trace.action_count += 1
            trace.movement_actions += 1
        observation = env._get_obs([pose], random_direction=False)[0]
        trace.online_observations += 1
        plan = {
            "modes": np.asarray([modes[slot] for slot in SLOT_TO_MODE]),
            "prior": np.zeros(8), "predicted_type": int(predicted_type),
        }
        _, evidence = select_from_current_rgb(
            observation, plan, semantic, args, device, beta=0.0)
        evidence_sum += evidence; evidence_count += 1
    return trace, selected_history, replans, evidence_count, committed_actions


def main():
    args = parse_args(); protocol = OfficialProtocol(); protocol.validate()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, inference, beliefs, modes, static = prepare(args, device)
    saved = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
    config = saved.get("config", {})
    scorer = MultiObjectiveTrajectoryScorer(
        route_channels=8,
        utility_scale=float(config.get("utility_score_scale", 1.0))).to(device)
    scorer.load_state_dict(saved["model"]); scorer.eval()
    diffusion_saved = torch.load(
        args.diffusion_checkpoint, map_location="cpu", weights_only=False)
    diffusion = ConditionalUnet1D(base=32).to(device)
    diffusion.load_state_dict(diffusion_saved["model"]); diffusion.eval()
    semantic = ProjectSemanticNet().to(device)
    semantic.load_state_dict(torch.load(
        args.semantic_checkpoint, map_location="cpu", weights_only=True)); semantic.eval()
    features = torch.load(args.type_features, map_location="cpu", weights_only=True)
    type_model = train_probe(
        features["text"]["train_seen"], features["text"]["val_seen"],
        args.seed, 40, device)
    with torch.inference_mode():
        predicted_type = type_model(
            features["text"][args.split]["features"].float().to(device)
        ).argmax(-1).cpu().numpy()
    beta = float(json.loads(args.rgb_results.read_text())[
        "selected_on_train_seen"]["rgb_weight"])
    env = CityNavBatch(
        args.split, environment_args(protocol, 1, args.max_episodes),
        batch_size=1, seed=0, rank=0, world_size=1)
    results, traces, histories, replan_counts, evidence_counts = {}, [], [], [], []
    committed_counts = []
    for index, _ in enumerate(env):
        observation = env._get_obs(random_direction=False)[0]
        trace, history, replans, evidence, committed = rollout_one(
            env, observation, index, modes[index], beliefs[index], inference,
            static, diffusion, scorer, semantic, predicted_type[index], beta,
            args, device, protocol)
        for item in attach_metric_only_ground_truth(env, [trace]):
            results[item["instr_id"]] = item
        traces.append(trace); histories.append(history)
        replan_counts.append(replans); evidence_counts.append(evidence)
        committed_counts.append(committed)
        if len(traces) >= len(rows): break
    metrics = env.eval_metrics(results)[0]
    result = {
        "experiment": "rolling anchored eight-trajectory RGB policy",
        "split": args.split, "count": len(results),
        "official_metrics": metrics,
        "rollout_diagnostics": summarize_traces(traces, 20),
        "policy_diagnostics": {
            "mean_replans": float(np.mean(replan_counts)),
            "mean_rgb_updates": float(np.mean(evidence_counts)),
            "mean_selected_route_changes": float(np.mean([
                sum(a != b for a, b in zip(value, value[1:]))
                for value in histories])),
            "mean_direct_commit_actions": float(np.mean(committed_counts)),
        },
        "anti_leakage": {
            "target_in_policy_or_rollout": False,
            "target_use": "attach_metric_only_ground_truth after rollout",
            "pointcloud_at_runtime": False,
            "test_unseen_loaded": False,
        },
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
