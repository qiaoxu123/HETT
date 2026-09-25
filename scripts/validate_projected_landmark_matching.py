#!/usr/bin/env python3
"""Gate test: project RGB building detections to named map landmarks."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.projection import (
    image_to_world,
    rasterize_polygon,
    transform_points,
    valid_polygon,
    view_corners,
    world_to_image_homography,
)
from independent_nav.semantic import TinyUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-visible-fraction", type=float, default=0.10)
    parser.add_argument("--min-component-pixels", type=int, default=8)
    parser.add_argument("--max-frames-per-split", type=int)
    return parser.parse_args()


def normalize_name(value):
    return "".join(value.lower().split())


def building_catalog(objects):
    catalogs, by_name = {}, {}
    for map_name, map_objects in objects.items():
        values = []
        names = defaultdict(list)
        for object_id, obj in map_objects.items():
            if obj.get("object_type") != "Building" or not obj.get("name"):
                continue
            entry = {
                "id": str(object_id),
                "name": obj["name"],
                "polygon": valid_polygon(obj["contour"]),
            }
            values.append(entry)
            names[normalize_name(obj["name"])].append(entry)
        catalogs[map_name] = values
        by_name[map_name] = {
            name: max(entries, key=lambda item: item["polygon"].area)
            for name, entries in names.items()
        }
    return catalogs, by_name


def nearest_identity(world_xy, catalog):
    point = Point(float(world_xy[0]), float(world_xy[1]))
    return min(catalog, key=lambda item: (
        item["polygon"].distance(point),
        item["polygon"].centroid.distance(point),
    ))["id"]


def wrong_pose_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["map_name"])].append(row)
    result = {}
    for values in groups.values():
        offset = max(1, len(values) // 2)
        for index, row in enumerate(values):
            replacement = values[(index + offset) % len(values)]
            result[row["image_id"]] = replacement
    return result


def infer(model, rows, image_root, resolution, batch_size, device):
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        images = []
        for row in batch_rows:
            image = cv2.imread(str(image_root / row["file_name"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_root / row["file_name"])
            image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
            images.append(torch.from_numpy(image[..., ::-1].copy()).permute(2, 0, 1))
        values = torch.stack(images).float().to(device).div(255.0)
        predictions.extend(model(values).argmax(1).cpu().numpy())
    return predictions


def summarize(records):
    coverage = np.asarray([row["coverage"] for row in records], dtype=np.float64)
    detected = coverage >= 0.25
    correct = np.asarray([row["correct_identity"] for row in records], dtype=bool)
    wrong = np.asarray([row["wrong_pose_correct_identity"] for row in records], dtype=bool)
    wrong_name_valid = np.asarray(
        [row["wrong_name_correct_identity"] is not None for row in records], dtype=bool)
    wrong_name = np.asarray(
        [bool(row["wrong_name_correct_identity"]) for row in records], dtype=bool)
    errors = np.asarray([row["centroid_error_m"] for row in records], dtype=np.float64)
    roundtrip = np.asarray([row["roundtrip_error_m"] for row in records], dtype=np.float64)
    return {
        "queries": len(records),
        "mean_visible_landmark_coverage_percent": float(coverage.mean() * 100),
        "median_visible_landmark_coverage_percent": float(np.median(coverage) * 100),
        "detection_rate_coverage_10_percent": float((coverage >= 0.10).mean() * 100),
        "detection_rate_coverage_25_percent": float(detected.mean() * 100),
        "detection_rate_coverage_50_percent": float((coverage >= 0.50).mean() * 100),
        "named_top1_given_detected": float(correct[detected].mean() * 100) if detected.any() else 0.0,
        "end_to_end_named_top1": float((correct & detected).mean() * 100),
        "wrong_pose_named_top1_given_detected": (
            float(wrong[detected].mean() * 100) if detected.any() else 0.0),
        "wrong_name_named_top1_given_detected": (
            float(wrong_name[detected & wrong_name_valid].mean() * 100)
            if (detected & wrong_name_valid).any() else 0.0),
        "mean_projected_component_centroid_error_m": (
            float(np.nanmean(errors[detected])) if detected.any() else None),
        "median_projected_component_centroid_error_m": (
            float(np.nanmedian(errors[detected])) if detected.any() else None),
        "geometry_roundtrip_mean_error_m": float(roundtrip.mean()),
        "geometry_roundtrip_max_error_m": float(roundtrip.max()),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    catalogs, names = building_catalog(objects)
    all_rows = [json.loads(line) for line in open(args.benchmark_dir / "manifest.jsonl")]
    rows = [row for row in all_rows if abs(row["altitude_m"] - args.altitude) < 1e-6]
    if args.max_frames_per_split:
        limited = []
        for split in ("val_seen", "val_unseen"):
            limited.extend([row for row in rows if row["split"] == split][
                :args.max_frames_per_split])
        rows = limited
    wrong_rows = wrong_pose_rows(rows)
    predictions = infer(
        model, rows, args.benchmark_dir / "images", args.resolution,
        args.batch_size, device)

    records = []
    examples = []
    for row, prediction in zip(rows, predictions):
        corners = view_corners(row["x"], row["y"], row["yaw_rad"], row["altitude_m"])
        homography = world_to_image_homography(corners, args.resolution)
        wrong_row = wrong_rows[row["image_id"]]
        wrong_corners = view_corners(
            wrong_row["x"], wrong_row["y"], wrong_row["yaw_rad"], row["altitude_m"])
        wrong_homography = world_to_image_homography(wrong_corners, args.resolution)
        view_polygon = valid_polygon(corners)
        building_mask = prediction == 1
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            building_mask.astype(np.uint8), connectivity=8)

        for name, visible_fraction in zip(
                row["resolved_landmarks"], row["landmark_visible_fractions"]):
            if visible_fraction < args.min_visible_fraction:
                continue
            query = names[row["map_name"]].get(normalize_name(name))
            if query is None:
                continue
            visible = query["polygon"].intersection(view_polygon)
            gt_mask = rasterize_polygon(visible, homography, args.resolution).astype(bool)
            if not gt_mask.any():
                continue
            coverage = float((building_mask & gt_mask).sum() / gt_mask.sum())
            best_label, best_overlap = None, 0
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] < args.min_component_pixels:
                    continue
                overlap = int(((labels == component) & gt_mask).sum())
                if overlap > best_overlap:
                    best_label, best_overlap = component, overlap

            correct_identity = wrong_identity = False
            wrong_name_identity = None
            centroid_error = float("nan")
            if best_label is not None:
                pixel = centroids[best_label][None]
                world = image_to_world(pixel, homography)[0]
                wrong_world = image_to_world(pixel, wrong_homography)[0]
                correct_identity = nearest_identity(world, catalogs[row["map_name"]]) == query["id"]
                wrong_identity = nearest_identity(wrong_world, catalogs[row["map_name"]]) == query["id"]
                alternatives = [item for item in catalogs[row["map_name"]]
                                if item["id"] != query["id"]]
                if alternatives:
                    replacement = min(
                        alternatives,
                        key=lambda item: item["polygon"].centroid.distance(query["polygon"].centroid))
                    wrong_name_identity = (
                        nearest_identity(world, catalogs[row["map_name"]]) == replacement["id"])
                target_centroid = np.asarray(
                    [visible.centroid.x, visible.centroid.y], dtype=np.float64)
                centroid_error = float(np.linalg.norm(world - target_centroid))

            target_centroid = np.asarray([[visible.centroid.x, visible.centroid.y]])
            roundtrip = image_to_world(
                transform_points(target_centroid, homography), homography)[0]
            roundtrip_error = float(np.linalg.norm(roundtrip - target_centroid[0]))
            record = {
                "split": row["split"], "image_id": row["image_id"],
                "map_name": row["map_name"], "landmark_name": name,
                "visible_fraction": visible_fraction, "coverage": coverage,
                "correct_identity": bool(correct_identity),
                "wrong_pose_correct_identity": bool(wrong_identity),
                "wrong_name_correct_identity": (
                    bool(wrong_name_identity) if wrong_name_identity is not None else None),
                "centroid_error_m": centroid_error,
                "roundtrip_error_m": roundtrip_error,
            }
            records.append(record)
            if len(examples) < 12 and coverage >= 0.25:
                image = cv2.imread(str(
                    args.benchmark_dir / "images" / row["file_name"]), cv2.IMREAD_COLOR)
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
                pred = cv2.resize(
                    building_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
                overlay = image.copy()
                overlay[pred.astype(bool)] = (
                    0.55 * overlay[pred.astype(bool)] + 0.45 * np.asarray([0, 0, 255])
                ).astype(np.uint8)
                cv2.putText(overlay, f"{name[:24]} cov={coverage:.2f}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                examples.append(overlay)

    result = {
        "config": {
            "altitude_m": args.altitude,
            "resolution": args.resolution,
            "min_visible_fraction": args.min_visible_fraction,
            "detection_threshold_coverage": 0.25,
            "visual_model": "TinyUNet trained on train_seen map classes",
            "camera": "top-down orthomosaic crop with exact pose homography",
        },
        "anti_leakage": {
            "target_coordinate": False,
            "target_identity": False,
            "matching_inputs": "RGB semantic mask, UAV pose/altitude/yaw, named map polygons",
            "landmark_polygons": "runtime map input and evaluation reference",
            "wrong_pose_control": "same RGB component projected with another pose from the same map",
        },
        "splits": {
            split: summarize([record for record in records if record["split"] == split])
            for split in ("val_seen", "val_unseen")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with open(args.output_dir / "records.jsonl", "w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    if examples:
        rows_visual = []
        for start in range(0, len(examples), 4):
            values = examples[start:start + 4]
            values.extend([np.zeros_like(values[0])] * (4 - len(values)))
            rows_visual.append(np.concatenate(values, axis=1))
        cv2.imwrite(str(args.output_dir / "examples.jpg"), np.concatenate(rows_visual, axis=0))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
