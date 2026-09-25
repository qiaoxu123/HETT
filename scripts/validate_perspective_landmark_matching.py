#!/usr/bin/env python3
"""Validate RGB-to-world-to-named-landmark matching on perspective PLY views."""

import argparse
import json
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely import contains_xy
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.pointcloud import unproject_depth
from independent_nav.projection import valid_polygon
from independent_nav.semantic import ProjectSemanticNet


def normalize_name(value):
    return "".join(value.lower().split())


def build_catalog(objects):
    catalogs, groups = {}, {}
    for map_name, map_objects in objects.items():
        values, names = [], defaultdict(list)
        for object_id, obj in map_objects.items():
            if obj.get("object_type") != "Building" or not obj.get("name"):
                continue
            entry = {"id": str(object_id), "name": obj["name"],
                     "polygon": valid_polygon(obj["contour"])}
            values.append(entry)
            names[normalize_name(obj["name"])].append(entry)
        catalogs[map_name] = values
        groups[map_name] = names
    return catalogs, groups


def resolve(query, groups):
    key = normalize_name(query)
    if key in groups:
        return max(groups[key], key=lambda item: item["polygon"].area)
    ranked = sorted((SequenceMatcher(None, key, name).ratio(), name)
                    for name in groups)
    if not ranked or ranked[-1][0] < 0.58:
        return None
    return max(groups[ranked[-1][1]], key=lambda item: item["polygon"].area)


def component_identity(world_xy, catalog):
    """Bind a projected component by polygon support, falling back to distance."""
    support = np.asarray([
        contains_xy(item["polygon"], world_xy[:, 0], world_xy[:, 1]).sum()
        for item in catalog
    ])
    if support.max(initial=0) > 0:
        return catalog[int(support.argmax())]["id"]
    center = np.median(world_xy, axis=0)
    point = Point(float(center[0]), float(center[1]))
    return min(catalog, key=lambda item: (
        item["polygon"].distance(point),
        item["polygon"].centroid.distance(point)))["id"]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--min-visible-pixels", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectSemanticNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    processed = json.load(open(args.data_root / "cityrefer/processed_descriptions.json"))
    catalogs, groups = build_catalog(objects)
    trajectories = json.load(open(
        args.data_root / "processed_citynav" / f"citynav_{args.split}.json"))
    by_map = defaultdict(list)
    for row in trajectories:
        by_map[f'{row["area"]}_block_{row["block"]}'].append(row)
    manifest = [json.loads(line) for line in
                (args.dataset / "manifest.jsonl").read_text().splitlines()]
    manifest = [row for row in manifest if row["split"] == args.split]

    records = []
    for row in manifest:
        values = np.load(args.dataset / row["file"])
        image = torch.from_numpy(values["rgb"].copy()).permute(2, 0, 1)[None]
        prediction = model(image.float().to(device).div(255.0)).argmax(1)[0].cpu().numpy()
        valid = values["valid"].astype(bool)
        depth = values["depth"].astype(np.float32)
        world = unproject_depth(depth, row["pose"], args.hfov)
        episode = by_map[row["map_name"]][row["episode_index_within_map"]]
        object_id, ann_id = str(episode["object_ids"][0]), int(episode["ann_ids"][0])
        annotations = processed.get(row["map_name"], {}).get(object_id, [])
        queries = annotations[ann_id].get("landmarks", []) if ann_id < len(annotations) else []
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            ((prediction == 1) & valid).astype(np.uint8), connectivity=8)
        for query_name in queries:
            landmark = resolve(query_name, groups[row["map_name"]])
            if landmark is None:
                continue
            inside = np.zeros(valid.shape, dtype=bool)
            inside[valid] = contains_xy(
                landmark["polygon"], world[..., 0][valid], world[..., 1][valid])
            true_pixels = inside & (values["semantic"] == 1)
            visible_pixels = int(true_pixels.sum())
            if visible_pixels < args.min_visible_pixels:
                continue
            coverage = float(((prediction == 1) & true_pixels).sum() / visible_pixels)
            best_label, best_overlap = None, 0
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] < 4:
                    continue
                overlap = int(((labels == component) & true_pixels).sum())
                if overlap > best_overlap:
                    best_label, best_overlap = component, overlap
            correct, error = False, float("nan")
            if best_label is not None:
                component_world = world[(labels == best_label) & valid, :2]
                center = np.median(component_world, axis=0)
                correct = component_identity(
                    component_world, catalogs[row["map_name"]]) == landmark["id"]
                error = float(np.linalg.norm(center - np.asarray([
                    landmark["polygon"].centroid.x, landmark["polygon"].centroid.y])))
            records.append({"coverage": coverage, "correct": bool(correct),
                            "centroid_error_m": error, "visible_pixels": visible_pixels})

    coverage = np.asarray([record["coverage"] for record in records])
    detected = coverage >= 0.25
    correct = np.asarray([record["correct"] for record in records])
    errors = np.asarray([record["centroid_error_m"] for record in records])
    result = {
        "split": args.split, "frames": len(manifest), "landmark_queries": len(records),
        "detection_rate_coverage_25_percent": float(detected.mean() * 100),
        "named_top1_given_detected": float(correct[detected].mean() * 100),
        "end_to_end_named_top1": float((correct & detected).mean() * 100),
        "median_component_to_landmark_centroid_error_m": float(np.nanmedian(errors[detected])),
        "anti_leakage": {
            "target_coordinates_or_identity": False,
            "frame_selection": "trajectory pose only",
            "ground_truth_semantics": "visibility and metrics only",
            "model_inputs": "rendered RGB only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
