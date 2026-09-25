#!/usr/bin/env python3
"""Build target-coordinate-free RGB sequences for temporal belief training.

Frame selection uses only trajectory order.  Target coordinates and object
identity are not read by this builder, and no target marker is rendered.
Supervision is joined later from the split row under a labels-only interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_altitude_view_benchmark import render_crop, view_corners
from multiagent.mapdata import MAP_BOUNDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/temporal_belief_rgb_gate1"))
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=128)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view-source", choices=("trajectory", "candidates"),
                        default="candidates")
    parser.add_argument("--candidate-dir", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/"
        "runs/candidate_rgb_data/geometry_top5_oof_grouped_enriched"))
    return parser.parse_args()


def map_name(row):
    return f'{row["area"]}_block_{row["block"]}'


def balanced_indices(rows, count, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[map_name(row)].append(index)
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


def pose_xy_yaw(raw):
    if len(raw) == 6:
        return float(raw[0]), float(raw[1]), math.atan2(raw[4], raw[3])
    if len(raw) == 5:
        return float(raw[0]), float(raw[1]), float(raw[3])
    raise ValueError(f"unsupported pose length {len(raw)}")


def temporal_indices(length, frames):
    """Choose trajectory-order quantiles without target position or distance."""
    if length <= 0:
        return []
    count = min(length, frames)
    return sorted(set(np.rint(np.linspace(0, length - 1, count)).astype(int).tolist()))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for value in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(value)
    return digest.hexdigest()


def normalized_to_world(point, name):
    bounds = MAP_BOUNDS[name]
    return (bounds.x_min + float(point[0]) * 410.0,
            bounds.y_max - float(point[1]) * 410.0)


def candidate_views(candidate_row, name, frames):
    centers = candidate_row.get("landmark_centers", [])
    landmark_world = [normalized_to_world(center, name) for center in centers]
    output = []
    for rank, point in enumerate(candidate_row["candidates"][:frames]):
        x, y = normalized_to_world(point, name)
        if landmark_world:
            center = np.asarray(landmark_world).mean(0)
            yaw = math.atan2(center[1] - y, center[0] - x)
        else:
            yaw = 0.0
        output.append((rank, x, y, yaw))
    return output


def build_split(args, split, count, seed):
    source_path = args.data_root / "processed_citynav" / f"citynav_{split}.json"
    rows = json.loads(source_path.read_text())
    candidate_path = args.candidate_dir / f"{split}.jsonl"
    candidates = [json.loads(line) for line in candidate_path.read_text().splitlines()
                  if line.strip()]
    if len(candidates) != len(rows):
        raise ValueError(f"candidate alignment mismatch for {split}")
    selected = balanced_indices(rows, count, seed)
    image_dir = args.output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    cache, output = {}, []
    for episode_id, source_row in enumerate(selected):
        row = rows[source_row]
        name = map_name(row)
        if (candidates[source_row]["map_name"] != name
                or candidates[source_row]["description"] != row["descriptions"][0]):
            raise ValueError(f"candidate semantic mismatch at {split}:{source_row}")
        if name not in cache:
            image_path = args.data_root / "rgbd" / f"{name}.png"
            tif_path = args.data_root / "rgbd" / f"{name}.tif"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            raster = rasterio.open(tif_path)
            cache[name] = image, raster
        image, raster = cache[name]
        frames = []
        if args.view_source == "trajectory":
            views = [(step, *pose_xy_yaw(row["trajectory"][step]))
                     for step in temporal_indices(len(row["trajectory"]), args.frames)]
            selection_inputs = "trajectory length and frame order only"
        else:
            views = candidate_views(candidates[source_row], name, args.frames)
            selection_inputs = "predicted candidate positions and named-landmark centers only"
        for order, (source_index, x, y, yaw) in enumerate(views):
            crop = render_crop(
                image, raster, view_corners(x, y, yaw, args.altitude), args.image_size)
            file_name = f"e{episode_id:05d}_t{order:02d}.jpg"
            if not cv2.imwrite(str(image_dir / file_name), crop,
                               [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"failed to write {file_name}")
            frames.append({
                "order": order,
                "trajectory_step": (source_index
                                    if args.view_source == "trajectory" else None),
                "source_view_index": source_index,
                "x": x, "y": y, "yaw_rad": yaw,
                "altitude_m": args.altitude,
                "file": str(Path("images") / split / file_name),
            })
        output.append({
            "split": split, "episode_id": episode_id,
            "source_row": source_row, "map_name": name,
            "frames": frames,
            "view_source": args.view_source,
            "selection_inputs": selection_inputs,
            "contains_target_coordinate": False,
            "contains_target_object_identity": False,
        })
    for _, raster in cache.values():
        raster.close()
    return source_path, output


def main():
    args = parse_args()
    if args.frames < 1:
        raise ValueError("frames must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests, sources = {}, {}
    for split, count, offset in (
            ("train_seen", args.train_count, 0),
            ("val_seen", args.val_count, 1009)):
        source, rows = build_split(args, split, count, args.seed + offset)
        manifests[split], sources[split] = rows, source
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {
        "counts": {split: len(rows) for split, rows in manifests.items()},
        "frames": {split: sum(len(row["frames"]) for row in rows)
                   for split, rows in manifests.items()},
        "selection": ("map-balanced episodes; "
                      + ("trajectory-order quantile frames" if args.view_source == "trajectory"
                         else "predicted candidate-centered views")),
        "anti_leakage": {
            "frame_selection_reads_target_coordinate": False,
            "frame_selection_reads_target_identity": False,
            "render_contains_target_marker": False,
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
            "human_trajectory_limit": (
                "applies only in trajectory mode; candidate mode does not use trajectory poses"),
        },
        "source_sha256": {split: sha256(path) for split, path in sources.items()},
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
