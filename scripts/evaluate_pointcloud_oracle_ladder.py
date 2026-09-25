#!/usr/bin/env python3
"""Measure the landmark-binding ceiling from point-cloud teacher channels."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely import contains_xy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.pointcloud import unproject_depth
from independent_nav.semantic import ProjectSemanticDepthNet, ProjectSemanticNet
from scripts.validate_perspective_landmark_matching import (
    build_catalog, component_identity, resolve,
)


def rotated_rows(rows):
    by_map = defaultdict(list)
    for row in rows:
        by_map[row["map_name"]].append(row)
    result = {}
    for values in by_map.values():
        offset = max(1, len(values) // 2)
        for index, row in enumerate(values):
            result[row["frame_id"]] = values[(index + offset) % len(values)]
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--min-visible-pixels", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectSemanticNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    depth_model = None
    if args.depth_checkpoint is not None:
        depth_model = ProjectSemanticDepthNet().to(device)
        depth_model.load_state_dict(torch.load(
            args.depth_checkpoint, map_location="cpu", weights_only=True))
        depth_model.eval()
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
    wrong_rows = rotated_rows(manifest)

    records = []
    for row in manifest:
        values = np.load(args.dataset / row["file"])
        image = torch.from_numpy(values["rgb"].copy()).permute(2, 0, 1)[None]
        prediction = model(image.float().to(device).div(255.0)).argmax(1)[0].cpu().numpy()
        predicted_depth = None
        depth_prediction = None
        if depth_model is not None:
            depth_semantic, normalized_depth = depth_model.predict_all(
                image.float().to(device).div(255.0))
            depth_prediction = depth_semantic.argmax(1)[0].cpu().numpy()
            predicted_depth = np.expm1(
                normalized_depth[0].cpu().numpy() * np.log1p(260.0))
        valid = values["valid"].astype(bool)
        depth = values["depth"].astype(np.float32)
        world = unproject_depth(depth, row["pose"], args.hfov)
        predicted_world = (unproject_depth(predicted_depth, row["pose"], args.hfov)
                           if predicted_depth is not None else None)
        wrong_predicted_world = (unproject_depth(
            predicted_depth, wrong_rows[row["frame_id"]]["pose"], args.hfov)
            if predicted_depth is not None else None)
        wrong_world = unproject_depth(
            depth, wrong_rows[row["frame_id"]]["pose"], args.hfov)
        episode = by_map[row["map_name"]][row["episode_index_within_map"]]
        object_id, ann_id = str(episode["object_ids"][0]), int(episode["ann_ids"][0])
        annotations = processed.get(row["map_name"], {}).get(object_id, [])
        queries = annotations[ann_id].get("landmarks", []) if ann_id < len(annotations) else []
        predicted_building = (prediction == 1) & valid
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            predicted_building.astype(np.uint8), connectivity=8)
        for query_name in queries:
            landmark = resolve(query_name, groups[row["map_name"]])
            if landmark is None:
                continue
            inside = np.zeros(valid.shape, dtype=bool)
            inside[valid] = contains_xy(
                landmark["polygon"], world[..., 0][valid], world[..., 1][valid])
            true_pixels = inside & (values["semantic"] == 1)
            visible = int(true_pixels.sum())
            if visible < args.min_visible_pixels:
                continue
            pixel_coverage = float((predicted_building & true_pixels).sum() / visible)
            wrong_inside = np.zeros(valid.shape, dtype=bool)
            wrong_inside[valid] = contains_xy(
                landmark["polygon"], wrong_world[..., 0][valid], wrong_world[..., 1][valid])
            wrong_coverage = float((predicted_building & wrong_inside).sum() / visible)

            best_label, best_overlap = None, 0
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] < 4:
                    continue
                overlap = int(((labels == component) & true_pixels).sum())
                if overlap > best_overlap:
                    best_label, best_overlap = component, overlap
            component_correct = False
            if best_label is not None and pixel_coverage >= 0.25:
                component_world = world[(labels == best_label) & valid, :2]
                component_correct = component_identity(
                    component_world, catalogs[row["map_name"]]) == landmark["id"]
            predicted_depth_coverage = None
            buffer_coverages = None
            wrong_pose_buffer_coverages = None
            wrong_name_buffer_coverages = None
            if predicted_world is not None:
                predicted_inside = np.zeros(valid.shape, dtype=bool)
                predicted_inside[valid] = contains_xy(
                    landmark["polygon"], predicted_world[..., 0][valid],
                    predicted_world[..., 1][valid])
                predicted_depth_coverage = float(
                    (((depth_prediction == 1) & valid & predicted_inside).sum()) / visible)
                alternatives = [item for item in catalogs[row["map_name"]]
                                if item["id"] != landmark["id"]]
                wrong_landmark = min(
                    alternatives,
                    key=lambda item: item["polygon"].distance(landmark["polygon"]))
                buffer_coverages = {}
                wrong_pose_buffer_coverages = {}
                wrong_name_buffer_coverages = {}
                for radius in (0, 5, 10, 20):
                    polygon = landmark["polygon"].buffer(radius)
                    wrong_polygon = wrong_landmark["polygon"].buffer(radius)
                    correct_mask = np.zeros(valid.shape, dtype=bool)
                    wrong_pose_mask = np.zeros(valid.shape, dtype=bool)
                    wrong_name_mask = np.zeros(valid.shape, dtype=bool)
                    correct_mask[valid] = contains_xy(
                        polygon, predicted_world[..., 0][valid],
                        predicted_world[..., 1][valid])
                    wrong_pose_mask[valid] = contains_xy(
                        polygon, wrong_predicted_world[..., 0][valid],
                        wrong_predicted_world[..., 1][valid])
                    wrong_name_mask[valid] = contains_xy(
                        wrong_polygon, predicted_world[..., 0][valid],
                        predicted_world[..., 1][valid])
                    for destination, mask in (
                            (buffer_coverages, correct_mask),
                            (wrong_pose_buffer_coverages, wrong_pose_mask),
                            (wrong_name_buffer_coverages, wrong_name_mask)):
                        destination[str(radius)] = float(
                            (((depth_prediction == 1) & valid & mask).sum()) / visible)
            records.append({
                "predicted_pixel_coverage": pixel_coverage,
                "wrong_pose_pixel_coverage": wrong_coverage,
                "component_correct": component_correct,
                "predicted_depth_pixel_coverage": predicted_depth_coverage,
                "predicted_depth_buffer_coverages": buffer_coverages,
                "wrong_pose_predicted_depth_buffer_coverages": wrong_pose_buffer_coverages,
                "wrong_name_predicted_depth_buffer_coverages": wrong_name_buffer_coverages,
            })

    predicted = np.asarray([row["predicted_pixel_coverage"] for row in records]) >= 0.25
    wrong = np.asarray([row["wrong_pose_pixel_coverage"] for row in records]) >= 0.25
    component = np.asarray([row["component_correct"] for row in records])
    result = {
        "split": args.split,
        "frames": len(manifest),
        "visible_named_landmark_queries": len(records),
        "oracle_semantic_depth_landmark_id": {
            "end_to_end_confirmation": 100.0,
            "meaning": "ceiling on queries with valid teacher pixels",
        },
        "predicted_semantic_oracle_depth_pixel_binding": {
            "end_to_end_confirmation": float(predicted.mean() * 100),
            "meaning": "soft per-pixel polygon binding; no component identity bottleneck",
        },
        "predicted_semantic_oracle_depth_component_binding": {
            "end_to_end_confirmation": float(component.mean() * 100),
        },
        "wrong_pose_pixel_binding_control": {
            "end_to_end_confirmation": float(wrong.mean() * 100),
        },
        "anti_leakage": {
            "target_coordinates_or_identity": False,
            "teacher_semantic_and_depth": "metrics and oracle branches only",
            "predicted_branch_model_input": "RGB only",
            "wrong_pose": "another sampled pose from the same map",
        },
    }
    if depth_model is not None:
        predicted_depth_coverages = np.asarray([
            row["predicted_depth_pixel_coverage"] for row in records])
        predicted_depth_binding = predicted_depth_coverages >= 0.25
        result["predicted_semantic_predicted_depth_pixel_binding"] = {
            "end_to_end_confirmation": float(predicted_depth_binding.mean() * 100),
            "meaning": "RGB-only semantic and depth projected into named map polygons",
            "coverage_percentiles": {
                str(percentile): float(np.percentile(predicted_depth_coverages, percentile))
                for percentile in (10, 25, 50, 75, 90)
            },
            "threshold_sweep": {
                str(threshold): float((predicted_depth_coverages >= threshold).mean() * 100)
                for threshold in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
            },
        }
        result["predicted_depth_spatial_tolerance_sweep"] = {}
        for radius in (0, 5, 10, 20):
            key = str(radius)
            correct_values = np.asarray([
                row["predicted_depth_buffer_coverages"][key] for row in records])
            wrong_pose_values = np.asarray([
                row["wrong_pose_predicted_depth_buffer_coverages"][key] for row in records])
            wrong_name_values = np.asarray([
                row["wrong_name_predicted_depth_buffer_coverages"][key] for row in records])
            result["predicted_depth_spatial_tolerance_sweep"][key] = {
                "correct_name_and_pose": float((correct_values >= 0.25).mean() * 100),
                "wrong_pose": float((wrong_pose_values >= 0.25).mean() * 100),
                "nearest_wrong_name": float((wrong_name_values >= 0.25).mean() * 100),
            }
        result["anti_leakage"]["predicted_depth_branch_model_input"] = "RGB only"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
