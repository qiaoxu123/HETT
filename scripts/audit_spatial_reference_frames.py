#!/usr/bin/env python3
"""Audit spatial-language reference frames and render RGB/landmark overlays."""

import argparse
import json
import math
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiagent.mapdata import MAP_BOUNDS
from scripts.audit_spatial_relation_probe import RELATION_PATTERNS, cue_names
from scripts.train_rgb_candidate_ranker import extract_crop
from scripts.validate_candidate_geometry import (
    mask_landmark_span,
    normalize_name,
    polygon_centroid,
    resolve_landmark_group,
)


TRANSFORMS = {
    "identity": np.array([[1, 0], [0, 1]], dtype=np.float32),
    "rotate_90": np.array([[0, -1], [1, 0]], dtype=np.float32),
    "rotate_180": np.array([[-1, 0], [0, -1]], dtype=np.float32),
    "rotate_270": np.array([[0, 1], [-1, 0]], dtype=np.float32),
    "mirror_x": np.array([[-1, 0], [0, 1]], dtype=np.float32),
    "mirror_y": np.array([[1, 0], [0, -1]], dtype=np.float32),
    "mirror_diagonal": np.array([[0, 1], [1, 0]], dtype=np.float32),
    "mirror_antidiagonal": np.array([[0, -1], [-1, 0]], dtype=np.float32),
}
CUE_VECTOR = {
    "left": np.array([-1.0, 0.0]), "right": np.array([1.0, 0.0]),
    "above": np.array([0.0, -1.0]), "below": np.array([0.0, 1.0]),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/spatial_reference_audit"))
    parser.add_argument("--minimum-axis-m", type=float, default=10.0)
    return parser.parse_args()


def normalize_xy(point, map_name):
    bounds = MAP_BOUNDS[map_name]
    return np.array([(point[0] - bounds.x_min) / 410.0,
                     (bounds.y_max - point[1]) / 410.0], dtype=np.float32)


def nearest_contour_delta(target, contours, map_name):
    points = np.concatenate([
        np.stack([normalize_xy(point, map_name) for point in contour])
        for contour in contours], axis=0)
    nearest = points[np.linalg.norm(points - target[None], axis=-1).argmin()]
    return target - nearest


def build_audit_examples(data_root, split, minimum_axis_m):
    objects = json.load(open(data_root / "cityrefer/objects.json"))
    processed = json.load(open(data_root / "cityrefer/processed_descriptions.json"))
    trajectories = json.load(open(
        data_root / f"processed_citynav/citynav_{split}.json"))
    examples = []
    cache = {}
    for item in trajectories:
        if item.get("dist_marker_to_target", 0.0) > 30.0:
            continue
        map_name = f'{item["area"]}_block_{item["block"]}'
        object_id = str(item["object_ids"][0])
        ann_id = int(item["ann_ids"][0])
        annotations = processed.get(map_name, {}).get(object_id, [])
        queries = annotations[ann_id].get("landmarks", []) if ann_id < len(annotations) else []
        if map_name not in cache:
            cache[map_name] = [obj for obj in objects[map_name].values() if obj.get("name")]
        target = normalize_xy(item["target_positions"][-1][:2], map_name)
        target_obj = objects[map_name][object_id]
        for query in queries[:9]:
            landmarks = resolve_landmark_group(query, cache[map_name])
            if not landmarks:
                continue
            name = landmarks[0]["name"]
            prompt, found = mask_landmark_span(item["descriptions"][0], query, name)
            cues = cue_names(prompt) if found else []
            if len(cues) != 1:
                continue
            contours = [landmark["contour"] for landmark in landmarks]
            centers = [polygon_centroid(contour) for contour in contours]
            areas = [abs(cv2.contourArea(np.asarray(contour, dtype=np.float32)))
                     for contour in contours]
            center_world = np.average(np.stack(centers), axis=0,
                                      weights=np.maximum(areas, 1e-6))
            center = normalize_xy(center_world, map_name)
            centroid_delta = target - center
            if min(abs(centroid_delta)) * 410.0 < minimum_axis_m:
                continue
            direction = np.asarray(item["trajectory"][0][3:5], dtype=np.float32)
            forward = np.array([direction[0], -direction[1]], dtype=np.float32)
            forward /= max(np.linalg.norm(forward), 1e-6)
            examples.append({
                "split": split, "map_name": map_name, "object_id": object_id,
                "description": item["descriptions"][0], "prompt": prompt,
                "cue": cues[0], "target": target, "center": center,
                "centroid_delta": centroid_delta,
                "boundary_delta": nearest_contour_delta(target, contours, map_name),
                "forward": forward, "landmark_name": name,
                "landmark_contours": contours,
                "target_contour": target_obj["contour"],
            })
    return examples


def transform_accuracy(examples, delta_key, matrix):
    correct = 0
    for row in examples:
        expected = matrix @ CUE_VECTOR[row["cue"]]
        correct += int(np.dot(row[delta_key], expected) > 0)
    return 100.0 * correct / max(len(examples), 1)


def agent_frame_accuracy(examples, delta_key):
    correct = 0
    for row in examples:
        forward = row["forward"]
        right = np.array([-forward[1], forward[0]])
        expected = {
            "left": -right, "right": right,
            "above": forward, "below": -forward,
        }[row["cue"]]
        correct += int(np.dot(row[delta_key], expected) > 0)
    return 100.0 * correct / max(len(examples), 1)


def summarize(examples):
    result = {"count": len(examples), "cue_counts": {}, "centroid": {}, "boundary": {}}
    for cue in CUE_VECTOR:
        result["cue_counts"][cue] = sum(row["cue"] == cue for row in examples)
    for delta_key, output_key in (("centroid_delta", "centroid"),
                                  ("boundary_delta", "boundary")):
        result[output_key]["transforms"] = {
            name: transform_accuracy(examples, delta_key, matrix)
            for name, matrix in TRANSFORMS.items()}
        result[output_key]["agent_heading"] = agent_frame_accuracy(examples, delta_key)
        by_map = defaultdict(list)
        for row in examples:
            by_map[row["map_name"]].append(row)
        oracle_correct = 0.0
        choices = {}
        for map_name, rows in by_map.items():
            scores = {name: transform_accuracy(rows, delta_key, matrix)
                      for name, matrix in TRANSFORMS.items()}
            choice = max(scores, key=scores.get)
            choices[map_name] = {"transform": choice, "accuracy": scores[choice],
                                 "count": len(rows)}
            oracle_correct += scores[choice] / 100.0 * len(rows)
        result[output_key]["per_map_oracle_accuracy"] = (
            100.0 * oracle_correct / max(len(examples), 1))
        result[output_key]["per_map_choices"] = choices
    return result


def world_to_crop(points, center_world, side_m, size=512):
    points = np.asarray(points, dtype=np.float32)
    half = side_m / 2.0
    x = (points[:, 0] - (center_world[0] - half)) / side_m * size
    y = ((center_world[1] + half) - points[:, 1]) / side_m * size
    return np.rint(np.stack((x, y), axis=-1)).astype(np.int32)


def render_example(row, data_root):
    map_name = row["map_name"]
    bounds = MAP_BOUNDS[map_name]
    target_world = np.array([bounds.x_min + row["target"][0] * 410.0,
                             bounds.y_max - row["target"][1] * 410.0])
    center_world = np.array([bounds.x_min + row["center"][0] * 410.0,
                             bounds.y_max - row["center"][1] * 410.0])
    crop_center = (target_world + center_world) / 2.0
    distance = np.linalg.norm(target_world - center_world)
    side_m = float(np.clip(distance + 100.0, 140.0, 320.0))
    rgb = cv2.imread(str(data_root / "rgbd" / f"{map_name}.png"), cv2.IMREAD_COLOR)
    with rasterio.open(data_root / "rgbd" / f"{map_name}.tif") as raster:
        transform = raster.transform
    image = cv2.cvtColor(extract_crop(rgb, transform, crop_center, side_m),
                         cv2.COLOR_RGB2BGR)
    overlay = image.copy()
    for contour in row["landmark_contours"]:
        cv2.fillPoly(overlay, [world_to_crop(contour, crop_center, side_m)], (255, 170, 0))
    cv2.addWeighted(overlay, 0.28, image, 0.72, 0, image)
    target_poly = world_to_crop(row["target_contour"], crop_center, side_m)
    cv2.polylines(image, [target_poly], True, (40, 220, 40), 4)
    landmark_px = world_to_crop([center_world], crop_center, side_m)[0]
    target_px = world_to_crop([target_world], crop_center, side_m)[0]
    cv2.arrowedLine(image, tuple(landmark_px), tuple(target_px), (0, 80, 255), 4,
                    tipLength=0.08)
    cv2.circle(image, tuple(landmark_px), 7, (255, 170, 0), -1)
    cv2.circle(image, tuple(target_px), 7, (0, 80, 255), -1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def render_montage(examples, data_root, output):
    chosen = []
    identity = TRANSFORMS["identity"]
    for cue in CUE_VECTOR:
        rows = [row for row in examples if row["cue"] == cue]
        good = [row for row in rows
                if np.dot(row["centroid_delta"], identity @ CUE_VECTOR[cue]) > 0]
        bad = [row for row in rows
               if np.dot(row["centroid_delta"], identity @ CUE_VECTOR[cue]) <= 0]
        chosen.extend((good[:1] + bad[:1]))
    figure, axes = plt.subplots(4, 2, figsize=(13, 18))
    for axis, row in zip(axes.flat, chosen):
        axis.imshow(render_example(row, data_root))
        agrees = np.dot(row["centroid_delta"], CUE_VECTOR[row["cue"]]) > 0
        axis.set_title(f'{row["cue"].upper()} | map-frame {"match" if agrees else "mismatch"}\n'
                       f'{row["landmark_name"]}', fontsize=11)
        axis.set_xlabel(textwrap.fill(row["description"], 58), fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Landmark mask (blue), target (green), landmark-to-target line (red)",
                    fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def serializable_examples(examples):
    result = []
    for row in examples:
        result.append({key: (value.tolist() if isinstance(value, np.ndarray) else value)
                       for key, value in row.items()
                       if key not in ("landmark_contours", "target_contour")})
    return result


def main():
    args = parse_args()
    by_split = {split: build_audit_examples(args.data_root, split, args.minimum_axis_m)
                for split in ("train_seen", "val_seen", "val_unseen")}
    result = {
        "config": {"minimum_axis_m": args.minimum_axis_m,
                   "patterns": {key: value.pattern for key, value in RELATION_PATTERNS.items()}},
        "splits": {split: summarize(rows) for split, rows in by_split.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "results.json", "w") as stream:
        json.dump(result, stream, indent=2)
    with open(args.output_dir / "val_unseen_examples.json", "w") as stream:
        json.dump(serializable_examples(by_split["val_unseen"]), stream, indent=2)
    render_montage(by_split["val_unseen"], args.data_root,
                   args.output_dir / "relation_overlay_examples.png")
    print(json.dumps({split: {
        "count": summary["count"],
        "centroid_transforms": summary["centroid"]["transforms"],
        "centroid_agent_heading": summary["centroid"]["agent_heading"],
        "centroid_per_map_oracle": summary["centroid"]["per_map_oracle_accuracy"],
    } for split, summary in result["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
