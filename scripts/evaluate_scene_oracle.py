#!/usr/bin/env python3
"""Measure map-scene oracle upper bounds for re-ranking Static B0 peaks."""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects  # noqa: E402
from multiagent.mapdata import MAP_BOUNDS  # noqa: E402
from multiagent.navigation_state import nms_topk_from_belief  # noqa: E402
from multiagent.scene_grounding.expected_scene import expected_scene  # noqa: E402
from multiagent.scene_grounding.oracle import oracle_scores  # noqa: E402
from multiagent.visual_attributes.labels import size_thresholds  # noqa: E402

KS = (1, 4, 8, 16); RADII = (20.0, 40.0)


def candidate_world(map_name, row, col, grid=30, map_meters=410.0):
    bounds = MAP_BOUNDS[map_name]
    return bounds.x_min + (col + 0.5) * map_meters / grid, bounds.y_max - (row + 0.5) * map_meters / grid


def metrics(records, variant, weight):
    result_rows = []
    for record in records:
        ranked = sorted(record["candidates"], key=lambda item: math.log(max(item["b0"], 1e-12)) + weight * item[variant], reverse=True)
        distances = np.asarray([item["goal_distance_m"] for item in ranked])
        value = {"top1_distance_m": float(distances[0]), "oracle_distance@8_m": float(distances[:8].min()),
                 "oracle_distance@16_m": float(distances[:16].min())}
        for k in KS:
            for radius in RADII: value[f"recall@{k}/{int(radius)}m"] = float(distances[:k].min() <= radius)
        result_rows.append(value)
    output = {"samples": len(result_rows)}
    for key in result_rows[0]: output[key] = float(np.mean([row[key] for row in result_rows]))
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--b0-cache-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if "test_unseen" in " ".join(map(str, vars(args).values())): raise ValueError("test_unseen is forbidden")
    objects = get_city_refer_objects(args.data_root / "cityrefer/objects.json", args.data_root / "cityrefer/processed_descriptions.json")
    train = json.loads((args.data_root / "processed_citynav/citynav_train_seen.json").read_text())
    train_targets = [objects[f"{row['area']}_block_{row['block']}"][int(row["object_ids"][0])] for row in train]
    thresholds = size_thresholds(train_targets)
    scene_cache = {}
    all_records = {}
    for split in ("val_seen", "val_unseen"):
        episodes = json.loads((args.data_root / "processed_citynav" / f"citynav_{split}.json").read_text())
        beliefs = torch.load(args.b0_cache_root / f"{split}_b0_eall.pt", map_location="cpu", weights_only=True)
        if len(episodes) != len(beliefs): raise ValueError(f"episode/cache mismatch for {split}")
        records = []
        for episode, belief in zip(episodes, beliefs):
            map_name = f"{episode['area']}_block_{episode['block']}"; map_objects = list(objects[map_name].values())
            target_xy = tuple(map(float, episode["target_positions"][-1][:2]))
            target_scene = expected_scene(target_xy, map_objects, size_thresholds=thresholds)
            target_obj = objects[map_name][int(episode["object_ids"][0])]
            target_scene["objects"].sort(key=lambda item: item["id"] != target_obj.id)
            ann = int(episode["ann_ids"][0]); landmark_names = target_obj.processed_descriptions[ann].landmarks if ann < len(target_obj.processed_descriptions) else []
            landmarks = [obj for obj in map_objects if obj.name in landmark_names]
            candidates = []
            for candidate in nms_topk_from_belief(belief, top_k=16, kernel_size=3):
                xy = candidate_world(map_name, candidate.row, candidate.col)
                key = (map_name, candidate.row, candidate.col)
                if key not in scene_cache: scene_cache[key] = expected_scene(xy, map_objects, size_thresholds=thresholds)
                score = oracle_scores(scene_cache[key], target_scene, xy, target_xy, landmarks)
                score.update({"b0": candidate.probability,
                              "goal_distance_m": float(np.linalg.norm(np.asarray(xy) - np.asarray(target_xy)))})
                candidates.append(score)
            records.append({"candidates": candidates})
        all_records[split] = records
    variants = ("attributes", "context", "geometry", "scene_geometry")
    weights = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    report = {"b0": {}, "oracles": {}, "test_unseen_read": False}
    for split in all_records: report["b0"][split] = metrics(all_records[split], "attributes", 0.0)
    for variant in variants:
        weight = max(weights, key=lambda value: metrics(all_records["val_seen"], variant, value)["recall@4/20m"])
        report["oracles"][variant] = {"weight": weight,
            "val_seen": metrics(all_records["val_seen"], variant, weight),
            "val_unseen": metrics(all_records["val_unseen"], variant, weight)}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value["val_unseen"] for key, value in report["oracles"].items()}, indent=2))


if __name__ == "__main__": main()
