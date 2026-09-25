#!/usr/bin/env python3
"""Evaluate independent relation goals with an independent action controller."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.controller import GeometricGoalController
from multiagent.mapdata import MAP_BOUNDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path,
                        default=Path("data/processed_citynav/citynav_val_unseen.json"))
    parser.add_argument("--goals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-meters", type=float, default=410.0)
    parser.add_argument("--success-radius", type=float, default=20.0)
    return parser.parse_args()


def episode_id(item):
    return [f'{item["area"]}_block_{item["block"]}',
            str(item["object_ids"][0]), int(item["ann_ids"][0])]


def unnormalize_goal(normalized, map_name, map_meters):
    bounds = MAP_BOUNDS[map_name]
    return (bounds.x_min + float(normalized[0]) * map_meters,
            bounds.y_max - float(normalized[1]) * map_meters)


def path_length(path):
    return sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))


def summarize(values):
    return {
        "count": len(values),
        "sr": 100.0 * float(np.mean([x["success"] for x in values])),
        "oracle_sr": 100.0 * float(np.mean([x["oracle_success"] for x in values])),
        "spl": 100.0 * float(np.mean([x["spl"] for x in values])),
        "mean_ne_m": float(np.mean([x["ne"] for x in values])),
        "median_ne_m": float(np.median([x["ne"] for x in values])),
        "mean_path_m": float(np.mean([x["path_length"] for x in values])),
        "mean_actions": float(np.mean([x["action_count"] for x in values])),
    }


def main():
    args = parse_args()
    payload = torch.load(args.goals, map_location="cpu")
    predicted = {tuple(key): goal.float().tolist()
                 for key, goal in zip(payload["episode_ids"], payload["goals"])}
    rows = json.loads(args.trajectories.read_text())
    rows = [row for row in rows if row.get("dist_marker_to_target", 0.0) <= 30.0]
    controller = GeometricGoalController()
    results = []

    for row in rows:
        key = tuple(episode_id(row))
        if key not in predicted:
            raise KeyError(f"missing predicted goal for {key}")
        map_name = key[0]
        goal_xy = unnormalize_goal(predicted[key], map_name, args.map_meters)
        pose = row["trajectory"][0]
        if len(pose) == 6:
            start_yaw = math.atan2(pose[4], pose[3])
        else:
            start_yaw = float(pose[3])
        plan = controller.plan((pose[0], pose[1]), start_yaw, goal_xy)
        target = tuple(map(float, row["target_positions"][-1][:2]))
        distances = [math.dist(point, target) for point in plan.path_xy]
        travelled = path_length(plan.path_xy)
        start_to_target = math.dist(plan.path_xy[0], target)
        success = distances[-1] <= args.success_radius
        results.append({
            "success": success,
            "oracle_success": min(distances) <= args.success_radius,
            "ne": distances[-1],
            "spl": float(success) * start_to_target /
                   max(travelled, start_to_target, 0.01),
            "path_length": travelled,
            "action_count": len(plan.actions),
        })

    if len(results) != len(predicted):
        raise RuntimeError(f"alignment mismatch: rows={len(results)} goals={len(predicted)}")
    output = {
        "scope": "HETT-independent predicted goal and action controller",
        "split": "val_unseen",
        "success_radius_m": args.success_radius,
        "hett_compatible_navigation_xy": summarize(results),
        "anti_leakage": {
            "controller_inputs": "start pose and independently predicted goal only",
            "target_coordinate_use": "metrics only, after trajectory generation",
            "hett_model": False,
            "hett_goal_candidates": False,
            "hett_action_generator": False,
            "height": "fixed 50 m; as in HETT, SR/SPL use horizontal XY distance",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
