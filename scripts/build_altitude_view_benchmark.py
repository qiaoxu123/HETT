#!/usr/bin/env python3
"""Build paired CityRefer views that differ only in camera altitude.

The benchmark deliberately uses validation splits only.  Each trajectory
contributes poses closest to fixed horizontal target-distance bands.  For
every pose, the script renders real orthomosaic crops at several altitudes
and a digital-zoom control derived from the highest-altitude crop.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pycocotools import mask as mask_utils
from shapely.geometry import Polygon


def normalize_name(value: str) -> str:
    return "".join(value.lower().split())


def valid_polygon(points) -> Polygon:
    polygon = Polygon(points)
    return polygon if polygon.is_valid else polygon.buffer(0)


def pose_xy_yaw(raw_pose):
    """Return x/y/yaw from CityNav's 5-D or direction-vector pose."""
    if len(raw_pose) == 5:
        return float(raw_pose[0]), float(raw_pose[1]), float(raw_pose[3])
    if len(raw_pose) == 6:
        return float(raw_pose[0]), float(raw_pose[1]), math.atan2(raw_pose[4], raw_pose[3])
    raise ValueError(f"Unsupported pose length: {len(raw_pose)}")


def view_corners(x: float, y: float, yaw: float, altitude: float) -> np.ndarray:
    front = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    left = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    center = np.asarray([x, y], dtype=np.float64)
    return np.asarray([
        center + altitude * (front + left),
        center + altitude * (front - left),
        center + altitude * (-front - left),
        center + altitude * (-front + left),
    ])


def balanced_sample(rows, count: int | None, seed: int):
    if count is None or count >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    by_map = defaultdict(list)
    for row in rows:
        by_map[f"{row['area']}_block_{row['block']}"] .append(row)
    for values in by_map.values():
        rng.shuffle(values)
    sampled = []
    map_names = sorted(by_map)
    while len(sampled) < count:
        added = False
        for map_name in map_names:
            if by_map[map_name] and len(sampled) < count:
                sampled.append(by_map[map_name].pop())
                added = True
        if not added:
            break
    return sampled


def select_distance_poses(row, distance_bands):
    target = np.asarray(row["target_positions"][-1][:2], dtype=np.float64)
    candidates = []
    for index, raw_pose in enumerate(row["trajectory"]):
        x, y, yaw = pose_xy_yaw(raw_pose)
        distance = float(np.linalg.norm(np.asarray([x, y]) - target))
        candidates.append((index, x, y, yaw, distance))
    selected = []
    used_indices = set()
    for band in distance_bands:
        candidate = min(candidates, key=lambda item: abs(item[-1] - band))
        if candidate[0] not in used_indices:
            selected.append((float(band), candidate))
            used_indices.add(candidate[0])
    return selected


def resolve_landmarks(map_objects, names):
    by_name = defaultdict(list)
    for obj in map_objects.values():
        if obj.get("name"):
            by_name[normalize_name(obj["name"])].append(obj)
    resolved, missing = [], []
    for name in names:
        matches = by_name.get(normalize_name(name), [])
        if matches:
            resolved.append(max(matches, key=lambda obj: valid_polygon(obj["contour"]).area))
        else:
            missing.append(name)
    return resolved, missing


def world_to_image_homography(corners, out_size: int):
    destination = np.asarray([
        [0, 0], [out_size - 1, 0],
        [out_size - 1, out_size - 1], [0, out_size - 1],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), destination)


def raster_to_image_homography(raster, corners, out_size: int):
    source = np.asarray(
        [[raster.index(float(x), float(y))[1], raster.index(float(x), float(y))[0]]
         for x, y in corners],
        dtype=np.float32,
    )
    destination = np.asarray([
        [0, 0], [out_size - 1, 0],
        [out_size - 1, out_size - 1], [0, out_size - 1],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(source, destination)


def render_crop(rgb, raster, corners, out_size: int):
    transform = raster_to_image_homography(raster, corners, out_size)
    return cv2.warpPerspective(rgb, transform, (out_size, out_size))


def rasterize_geometry(geometry, homography, out_size: int):
    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    if geometry.is_empty:
        return mask
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for polygon in polygons:
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        points = np.asarray(polygon.exterior.coords[:-1], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points[None, :, :], homography)[0]
        pixels = np.round(transformed).astype(np.int32)
        if len(pixels) >= 3:
            cv2.fillPoly(mask, [pixels], 1)
    return mask


def digital_zoom(high_image, low_altitude: float, high_altitude: float):
    height, width = high_image.shape[:2]
    scale = low_altitude / high_altitude
    crop_h = max(2, int(round(height * scale)))
    crop_w = max(2, int(round(width * scale)))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    crop = high_image[top:top + crop_h, left:left + crop_w]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)


def image_comparison(real, control):
    delta = real.astype(np.float32) - control.astype(np.float32)
    mse = float(np.mean(delta * delta))
    gray_real = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    gray_control = cv2.cvtColor(control, cv2.COLOR_BGR2GRAY)
    return {
        "mae": float(np.mean(np.abs(delta))),
        "psnr_db": float(99.0 if mse == 0 else 10 * math.log10(255.0**2 / mse)),
        "real_laplacian_variance": float(cv2.Laplacian(gray_real, cv2.CV_64F).var()),
        "zoom_laplacian_variance": float(cv2.Laplacian(gray_control, cv2.CV_64F).var()),
    }


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def summarize(rows, zoom_rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["distance_band_m"], row["altitude_m"])].append(row)
    by_condition = []
    for key, values in sorted(groups.items()):
        by_condition.append({
            "split": key[0],
            "distance_band_m": key[1],
            "altitude_m": key[2],
            "count": len(values),
            "target_visible_rate": 100 * np.mean([x["target_visible_fraction"] > 0 for x in values]),
            "target_fully_visible_rate": 100 * np.mean([x["target_visible_fraction"] >= 0.999 for x in values]),
            "mean_target_visible_fraction": 100 * np.mean([x["target_visible_fraction"] for x in values]),
            "mean_target_pixels": np.mean([x["target_mask_pixels"] for x in values]),
            "any_landmark_visible_rate": 100 * np.mean([x["any_landmark_visible"] for x in values]),
            "all_landmarks_visible_rate": 100 * np.mean([x["all_landmarks_visible"] for x in values]),
        })
    zoom_groups = defaultdict(list)
    for row in zoom_rows:
        zoom_groups[(row["split"], row["distance_band_m"], row["altitude_m"])].append(row)
    by_zoom = []
    for key, values in sorted(zoom_groups.items()):
        by_zoom.append({
            "split": key[0], "distance_band_m": key[1], "altitude_m": key[2],
            "count": len(values),
            "mean_mae": np.mean([x["mae"] for x in values]),
            "mean_psnr_db": np.mean([x["psnr_db"] for x in values]),
            "mean_real_laplacian_variance": np.mean([x["real_laplacian_variance"] for x in values]),
            "mean_zoom_laplacian_variance": np.mean([x["zoom_laplacian_variance"] for x in values]),
        })
    return {"samples": len(rows), "conditions": by_condition, "digital_zoom_controls": by_zoom}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("val_seen", "val_unseen"), default=("val_seen", "val_unseen"))
    parser.add_argument("--samples-per-split", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--altitudes", nargs="+", type=float, default=(30, 50, 60, 80, 120))
    parser.add_argument("--distance-bands", nargs="+", type=float, default=(80, 40, 20))
    parser.add_argument("--out-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def main():
    args = parse_args()
    if max(args.altitudes) <= 0 or min(args.altitudes) <= 0:
        raise ValueError("Altitudes must be positive AGL values")
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    image_root = output / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    objects = json.loads((repo / "data/cityrefer/objects.json").read_text())
    processed = json.loads((repo / "data/cityrefer/processed_descriptions.json").read_text())
    highest = max(args.altitudes)
    rows_out, zoom_out, annotations, zoom_annotations = [], [], [], []
    image_id = 0

    for split in args.splits:
        source = json.loads((repo / f"data/processed_citynav/citynav_{split}.json").read_text())
        source = balanced_sample(source, args.samples_per_split, args.seed)
        grouped = defaultdict(list)
        for row_index, row in enumerate(source):
            grouped[f"{row['area']}_block_{row['block']}"] .append((row_index, row))
        for map_name, map_rows in sorted(grouped.items()):
            rgb_path = repo / "data/rgbd" / f"{map_name}.png"
            tif_path = repo / "data/rgbd" / f"{map_name}.tif"
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(rgb_path)
            with rasterio.open(tif_path) as raster:
                for row_index, row in map_rows:
                    object_id = str(row["object_ids"][0])
                    ann_id = int(row["ann_ids"][0])
                    target = objects[map_name][object_id]
                    target_polygon = valid_polygon(target["contour"])
                    proc = processed.get(map_name, {}).get(object_id, [])[ann_id]
                    landmarks, missing_landmarks = resolve_landmarks(
                        objects[map_name], proc.get("landmarks", []),
                    )
                    instruction = row["descriptions"][0]
                    for distance_band, pose in select_distance_poses(row, args.distance_bands):
                        pose_index, x, y, yaw, actual_distance = pose
                        rendered, rendered_masks = {}, {}
                        base = f"{split}_{row_index:05d}_p{pose_index:03d}"
                        for altitude in sorted(args.altitudes):
                            corners = view_corners(x, y, yaw, altitude)
                            view_polygon = valid_polygon(corners)
                            target_intersection = target_polygon.intersection(view_polygon)
                            target_visible = target_intersection.area / max(target_polygon.area, 1e-9)
                            homography = world_to_image_homography(corners, args.out_size)
                            target_mask = rasterize_geometry(target_intersection, homography, args.out_size)
                            image = render_crop(rgb, raster, corners, args.out_size)
                            altitude_label = f"{altitude:g}m"
                            file_name = f"{base}_{altitude_label}.jpg"
                            relative_file = Path(split) / file_name
                            (image_root / split).mkdir(parents=True, exist_ok=True)
                            if not cv2.imwrite(
                                str(image_root / relative_file), image,
                                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                            ):
                                raise RuntimeError(f"Failed to write {relative_file}")
                            rendered[altitude] = image
                            rendered_masks[altitude] = target_mask
                            landmark_fractions = []
                            for landmark in landmarks:
                                polygon = valid_polygon(landmark["contour"])
                                landmark_fractions.append(
                                    polygon.intersection(view_polygon).area / max(polygon.area, 1e-9)
                                )
                            manifest = {
                                "image_id": image_id, "split": split, "map_name": map_name,
                                "source_row": row_index, "object_id": int(object_id), "ann_id": ann_id,
                                "pose_index": pose_index, "distance_band_m": distance_band,
                                "actual_target_distance_m": actual_distance,
                                "x": x, "y": y, "yaw_rad": yaw, "altitude_m": altitude,
                                "instruction": instruction, "processed_description": proc,
                                "resolved_landmarks": [item["name"] for item in landmarks],
                                "missing_landmarks": missing_landmarks,
                                "target_visible_fraction": float(target_visible),
                                "target_mask_pixels": int(target_mask.sum()),
                                "landmark_visible_fractions": landmark_fractions,
                                "any_landmark_visible": bool(landmark_fractions and max(landmark_fractions) > 0),
                                "all_landmarks_visible": bool(landmark_fractions and min(landmark_fractions) >= 0.999),
                                "file_name": str(relative_file),
                            }
                            rows_out.append(manifest)
                            annotations.append({
                                "split": split, "image_id": str(image_id), "sent": instruction,
                                "file_name": str(relative_file), "segmentation": encode_mask(target_mask),
                                "category_name": f"real_{split}_d{distance_band:g}_h{altitude:g}",
                                "sample_key": base, "modality": "real",
                                "distance_band_m": distance_band, "altitude_m": altitude,
                                "target_visible_fraction": float(target_visible),
                            })
                            image_id += 1
                        high_image = rendered[highest]
                        for altitude, real_image in rendered.items():
                            if altitude == highest:
                                continue
                            zoom_image = digital_zoom(high_image, altitude, highest)
                            zoom_name = f"{base}_{altitude:g}m_zoom_from_{highest:g}m.jpg"
                            zoom_relative = Path(split) / zoom_name
                            if not cv2.imwrite(
                                str(image_root / zoom_relative), zoom_image,
                                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                            ):
                                raise RuntimeError(f"Failed to write {zoom_relative}")
                            visible_fraction = next(
                                item["target_visible_fraction"] for item in reversed(rows_out)
                                if item["split"] == split and item["source_row"] == row_index
                                and item["pose_index"] == pose_index and item["altitude_m"] == altitude
                            )
                            zoom_annotations.append({
                                "split": split, "image_id": f"zoom_{image_id}", "sent": instruction,
                                "file_name": str(zoom_relative),
                                "segmentation": encode_mask(rendered_masks[altitude]),
                                "category_name": f"zoom_{split}_d{distance_band:g}_h{altitude:g}",
                                "sample_key": base, "modality": "zoom_from_high",
                                "distance_band_m": distance_band, "altitude_m": altitude,
                                "source_altitude_m": highest,
                                "target_visible_fraction": visible_fraction,
                            })
                            comparison = image_comparison(real_image, zoom_image)
                            zoom_out.append({
                                "split": split, "source_row": row_index, "pose_index": pose_index,
                                "distance_band_m": distance_band, "actual_target_distance_m": actual_distance,
                                "altitude_m": altitude, "source_altitude_m": highest, **comparison,
                            })

    with (output / "manifest.jsonl").open("w") as stream:
        for row in rows_out:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "annotations.jsonl").open("w") as stream:
        for row in annotations:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    fully_visible_real = [row for row in annotations if row["target_visible_fraction"] >= 0.999]
    fully_visible_zoom = [row for row in zoom_annotations if row["target_visible_fraction"] >= 0.999]
    visibility_by_key = defaultdict(list)
    for row in annotations:
        visibility_by_key[row["sample_key"]].append(row["target_visible_fraction"])
    common_keys = {
        key for key, values in visibility_by_key.items()
        if len(values) == len(args.altitudes) and min(values) >= 0.999
    }
    common_real = [row for row in annotations if row["sample_key"] in common_keys]
    common_zoom = [row for row in zoom_annotations if row["sample_key"] in common_keys]
    annotation_sets = {
        "annotations_real_full_visibility.jsonl": fully_visible_real,
        "annotations_zoom_full_visibility.jsonl": fully_visible_zoom,
        "annotations_semantic_common_visibility.jsonl": common_real + common_zoom,
    }
    for name, values in annotation_sets.items():
        with (output / name).open("w") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "digital_zoom.jsonl").open("w") as stream:
        for row in zoom_out:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows_out, zoom_out)
    summary["protocol"] = {
        "splits": list(args.splits), "samples_per_split": args.samples_per_split,
        "seed": args.seed, "altitudes_m": sorted(args.altitudes),
        "distance_bands_m": list(args.distance_bands), "out_size": args.out_size,
        "highest_altitude_for_digital_zoom_m": highest,
    }
    summary["semantic_annotation_counts"] = {
        "real_full_visibility": len(fully_visible_real),
        "zoom_full_visibility": len(fully_visible_zoom),
        "common_visibility_pose_count": len(common_keys),
        "common_visibility_real_and_zoom": len(common_real) + len(common_zoom),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
