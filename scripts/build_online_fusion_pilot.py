#!/usr/bin/env python3
"""Build a small causal online semantic/RGB fusion pilot without target input."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.official_protocol import OfficialProtocol, move_toward
from independent_nav.online_fusion_sample import (
    candidate_prior, previous_belief_reference, validate_episode_sample,
)
from independent_nav.projection import (
    transform_points, view_corners, world_to_image_homography,
)
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.mapdata import MAP_BOUNDS
from multiagent.observation import cropclient
from multiagent.space import Point2D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-dir", type=Path, default=Path(
            "../45-active-altitude/runs/candidate_rgb_data/"
            "geometry_top5_oof_grouped_enriched"
        )
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/online_fusion_pilot_step1")
    )
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def map_balanced_indices(rows, count, seed):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["map_name"]].append(index)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(count, len(rows)):
        changed = False
        for name in sorted(groups):
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
                changed = True
        if not changed:
            break
    return selected


def normalized_to_world(point, map_name, map_size_m=410.0):
    bounds = MAP_BOUNDS[map_name]
    return Point2D(
        bounds.x_min + float(point[0]) * map_size_m,
        bounds.y_max - float(point[1]) * map_size_m,
    )


def world_grid(map_name, grid_size, map_size_m=410.0):
    bounds = MAP_BOUNDS[map_name]
    axis = (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((
        bounds.x_min + xx.reshape(-1) * map_size_m,
        bounds.y_max - yy.reshape(-1) * map_size_m,
    ))


def frustum_indices(pose, map_name, grid_size, image_size, altitude_m):
    corners = view_corners(pose.x, pose.y, pose.yaw, altitude_m)
    homography = world_to_image_homography(corners, image_size)
    pixels = transform_points(world_grid(map_name, grid_size), homography)
    valid = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] <= image_size - 1)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] <= image_size - 1)
    )
    return np.flatnonzero(valid).astype(int).tolist()


def build_episode(row, trajectory, source_row, split, episode_id, args, protocol):
    if row["description"] != trajectory.descriptions[0]:
        raise ValueError(f"language alignment mismatch at {split}:{source_row}")
    expected_map = f"{trajectory.area}_block_{trajectory.block}"
    if row["map_name"] != expected_map:
        raise ValueError(f"map alignment mismatch at {split}:{source_row}")

    destination = normalized_to_world(row["candidates"][0], row["map_name"])
    pose = trajectory.start_pose
    image_dir = args.output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for step in range(protocol.max_action_len):
        rgb = cropclient.crop_image(
            row["map_name"], pose, (args.image_size, args.image_size), "rgb"
        )
        filename = f"e{episode_id:04d}_t{step:02d}.jpg"
        image_path = image_dir / filename
        if not cv2.imwrite(str(image_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {image_path}")
        visible = frustum_indices(
            pose, row["map_name"], args.grid_size, args.image_size,
            protocol.altitude_m,
        )
        frames.append({
            "step": step,
            "rgb_file": str(Path("images") / split / filename),
            "rgb_sha256": sha256(image_path),
            "pose": {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw},
            "remaining_budget": protocol.max_action_len - step,
            "previous_belief_ref": previous_belief_reference(step),
            "frustum_visible_indices": visible,
            # Without depth/occlusion verification, an in-frame non-detection
            # remains unknown and is never converted into negative evidence.
            "negative_evidence_allowed_indices": [],
            "occlusion_verified": False,
        })
        if pose.xy.dist_to(destination) <= 5.0:
            break
        pose = move_toward(pose, destination, protocol.move_iteration)

    return {
        "split": split,
        "episode_id": episode_id,
        "source_row": source_row,
        "map_name": row["map_name"],
        "instruction": row["description"],
        "landmark_names": row["landmark_names"],
        "relation_queries": row["relation_queries"],
        "landmark_centers": row["landmark_centers"],
        "landmark_boundaries": row["landmark_boundaries"],
        "ranked_candidates": row["candidates"],
        "initial_belief": candidate_prior(
            row["candidates"], args.grid_size).tolist(),
        "recurrent_state_contract": (
            "step 0 uses initial_belief; step t>0 uses fusion output at t-1"
        ),
        "frames": frames,
    }


def build_split(split, count, seed, args, protocol):
    candidate_path = args.candidate_dir / f"{split}.jsonl"
    rows = load_candidate_rows(candidate_path)
    trajectories = load_mturk_trajectories(split, "all", protocol.altitude_m)
    if len(rows) != len(trajectories):
        raise ValueError(f"candidate/trajectory count mismatch for {split}")
    selected = map_balanced_indices(rows, count, seed)
    episodes = [
        build_episode(rows[index], trajectories[index], index, split, episode_id,
                      args, protocol)
        for episode_id, index in enumerate(selected)
    ]
    path = args.output_dir / f"{split}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in episodes))
    return episodes, path


def main():
    args = parse_args()
    protocol = OfficialProtocol()
    protocol.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cropclient.load_image_cache()
    outputs = {}
    for split, count, offset in (
        ("train_seen", args.train_count, 0),
        ("val_seen", args.val_count, 1009),
    ):
        episodes, path = build_split(split, count, args.seed + offset, args, protocol)
        errors = [error for episode in episodes
                  for error in validate_episode_sample(episode)]
        rgb_files = [args.output_dir / frame["rgb_file"]
                     for episode in episodes for frame in episode["frames"]]
        rgb_hashes = [frame["rgb_sha256"]
                      for episode in episodes for frame in episode["frames"]]
        visible_masks = [frame["frustum_visible_indices"]
                         for episode in episodes for frame in episode["frames"]]
        outputs[split] = {
            "episodes": len(episodes),
            "frames": len(rgb_files),
            "minimum_frames": min(len(row["frames"]) for row in episodes),
            "maximum_frames": max(len(row["frames"]) for row in episodes),
            "all_rgb_readable": all(cv2.imread(str(path)) is not None for path in rgb_files),
            "unique_rgb_percent": 100.0 * len(set(rgb_hashes)) / max(len(rgb_hashes), 1),
            "episodes_with_changing_rgb": sum(
                len({frame["rgb_sha256"] for frame in row["frames"]}) > 1
                for row in episodes
            ),
            "all_frustum_masks_nonempty": all(bool(mask) for mask in visible_masks),
            "validation_errors": errors,
            "manifest_sha256": sha256(path),
        }
    gates = {
        "all_samples_valid": all(not value["validation_errors"] for value in outputs.values()),
        "all_rgb_readable": all(value["all_rgb_readable"] for value in outputs.values()),
        "current_rgb_changes_with_motion": all(
            value["episodes_with_changing_rgb"] == value["episodes"]
            for value in outputs.values()
        ),
        "all_frustum_masks_nonempty": all(
            value["all_frustum_masks_nonempty"] for value in outputs.values()
        ),
        "both_splits_nonempty": all(value["episodes"] > 0 for value in outputs.values()),
    }
    result = {
        "step": "Todo 1 online fusion sample pilot",
        "protocol": {
            "max_action_len": protocol.max_action_len,
            "move_iteration": protocol.move_iteration,
            "altitude_m": protocol.altitude_m,
        },
        "splits": outputs,
        "gates": gates,
        "passed": all(gates.values()),
        "visibility_policy": {
            "frustum": "pose-conditioned map cells",
            "occlusion": "unknown without depth teacher",
            "negative_evidence": "disabled unless occlusion is verified",
        },
        "anti_leakage": {
            "target_coordinate_loaded_into_sample": False,
            "target_object_identity_loaded_into_sample": False,
            "route_destination": "predicted rank-1 candidate only",
            "previous_belief": "causal recurrent reference; no target-derived history",
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
