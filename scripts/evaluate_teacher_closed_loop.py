#!/usr/bin/env python3
"""Offline closed-loop gate using learned views and teacher observations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.controller import GeometricGoalController
from independent_nav.runtime import (
    ActiveBeliefNavigator, ObservationEvidence, RuntimeConfig, ViewPlan,
)
from scripts.train_relation_active_view_head import (
    ViewValueHead, attach_per_landmark_cues, select_learned,
)
from scripts.evaluate_pointcloud_active_view_oracle import load_episode_metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path,
                        default=Path("runs/relation_active_view_gate10/records.pt"))
    parser.add_argument("--view-checkpoint", type=Path, default=Path(
        "runs/relation_active_view_gate11_per_landmark/model.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/teacher_closed_loop_gate12/results.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args()


def origin(record):
    normalized = record["candidates"][0]
    world = record["candidates_world"][0]
    return world[0] - normalized[0] * 410.0, world[1] + normalized[1] * 410.0


def normalize_world(point, map_origin):
    return ((float(point[0]) - map_origin[0]) / 410.0,
            (map_origin[1] - float(point[1])) / 410.0)


def belief_prior(record, grid_size=256):
    probability = torch.full((grid_size, grid_size), 1e-5)
    denominator = max(len(record["candidates"]) - 1, 1)
    for rank, candidate in enumerate(record["candidates"]):
        x = min(grid_size - 1, max(0, int(candidate[0] * grid_size)))
        y = min(grid_size - 1, max(0, int(candidate[1] * grid_size)))
        probability[y, x] = max(
            float(probability[y, x]),
            float(torch.sigmoid(torch.tensor(-0.3 * rank / denominator))))
    return probability


def observation(record, view, grid_size=256):
    signatures = record["signatures"][view]
    score = (2.0 * (0.5 * signatures[:, 0] + 0.5 * signatures[:, 1])
             + signatures[:, 5])
    evidence = torch.zeros(grid_size, grid_size)
    observable = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    for candidate, value in zip(record["candidates"], score):
        x = min(grid_size - 1, max(0, int(candidate[0] * grid_size)))
        y = min(grid_size - 1, max(0, int(candidate[1] * grid_size)))
        evidence[y, x] = float(value)
        observable[y, x] = True
    selected = int(np.argmax(score))
    return ObservationEvidence(
        evidence, observable, 1.0, f"teacher-view-{view}",
        float(np.clip(0.5 * signatures[selected, 0]
                      + 0.5 * signatures[selected, 1], 0.0, 1.0)),
        float(np.clip(signatures[selected, 3], 0.0, 1.0)))


def path_length(path):
    return sum(math.dist(left, right) for left, right in zip(path[:-1], path[1:]))


def evaluate_record(record, view):
    map_origin = origin(record)
    view_world = record["poses"][view, :2]
    view_xy = normalize_world(view_world, map_origin)
    descriptor = record["view_descriptors"][view]
    planner = lambda _belief, _xy, _alt, _count: ViewPlan(
        view_xy, descriptor[1], 0.0)
    runtime = ActiveBeliefNavigator(
        belief_prior(record), planner,
        RuntimeConfig(
            maximum_observations=1, entropy_view_threshold=0.0,
            concentration_threshold=0.15, stop_type_confidence=0.05,
            stop_relation_confidence=0.05, stable_goal_frames=2,
            stop_confirm_frames=2))
    start_xy = normalize_world(record["start"], map_origin)
    first = runtime.step(start_xy, 0.0, 50.0)
    controller = GeometricGoalController()
    start_world = tuple(record["start"])
    plan_to_view = controller.plan(start_world, 0.0, tuple(view_world))
    second = runtime.step(
        view_xy, plan_to_view.final_yaw, descriptor[1], observation(record, view))
    goal_world = (map_origin[0] + second.predicted_goal_xy[0] * 410.0,
                  map_origin[1] - second.predicted_goal_xy[1] * 410.0)
    plan_to_goal = controller.plan(tuple(view_world), plan_to_view.final_yaw, goal_world)
    goal_xy = second.predicted_goal_xy
    runtime.step(goal_xy, plan_to_goal.final_yaw, 50.0)
    final = runtime.step(goal_xy, plan_to_goal.final_yaw, 50.0)

    path = plan_to_view.path_xy + plan_to_goal.path_xy[1:]
    target_world = (map_origin[0] + record["target"][0] * 410.0,
                    map_origin[1] - record["target"][1] * 410.0)
    distances = [math.dist(point, target_world) for point in path]
    travelled = path_length(path)
    shortest = math.dist(path[0], target_world)
    success = final.should_stop and distances[-1] <= 20.0
    return {
        "success": success,
        "oracle_success": min(distances) <= 20.0,
        "correct_stop": final.should_stop,
        "ne": distances[-1],
        "spl": float(success) * shortest / max(shortest, travelled, 0.01),
        "path_length": travelled,
        "actions": len(plan_to_view.actions) + len(plan_to_goal.actions),
        "first_action": first.action,
        "post_observation_phase": second.phase.value,
    }


def summarize(rows):
    return {
        "count": len(rows),
        "sr": 100.0 * np.mean([row["success"] for row in rows]),
        "osr": 100.0 * np.mean([row["oracle_success"] for row in rows]),
        "correct_stop_rate": 100.0 * np.mean([row["correct_stop"] for row in rows]),
        "spl": 100.0 * np.mean([row["spl"] for row in rows]),
        "mean_ne_m": float(np.mean([row["ne"] for row in rows])),
        "mean_path_m": float(np.mean([row["path_length"] for row in rows])),
        "mean_actions": float(np.mean([row["actions"] for row in rows])),
    }


def main():
    args = parse_args()
    saved = torch.load(args.records, map_location="cpu", weights_only=False)
    records = saved["val_seen"]
    metadata = load_episode_metadata(args.data_root, "val_seen")
    attach_per_landmark_cues(records, metadata)
    checkpoint = torch.load(args.view_checkpoint, map_location="cpu", weights_only=True)
    model = ViewValueHead(checkpoint["input_dimension"])
    model.load_state_dict(checkpoint["model"]); model.eval()
    views = select_learned(model, records, torch.device("cpu"))
    rows = [evaluate_record(record, int(view))
            for record, view in zip(records, views)]
    result = {
        "val_seen_diagnostic": summarize(rows),
        "anti_leakage": {
            "runtime_inputs": (
                "initial relation candidates, learned view, teacher observation evidence, pose"),
            "target_coordinates": "metrics only after action generation",
            "target_in_runtime_or_stop": False,
            "view_head": "train_seen-trained, val_seen teacher-utility-selected",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
        },
        "scope": (
            "offline point-cloud-teacher closed-loop logic gate; not RGB or simulator navigation"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
