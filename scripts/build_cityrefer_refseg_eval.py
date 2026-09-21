#!/usr/bin/env python3
"""Build deterministic CityRefer oracle-ROI samples for RSRefSeg 2."""

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
from shapely.ops import nearest_points
from transformers import AutoProcessor


def normalize_name(value):
    return "".join(value.lower().split())


def polygon_area(points):
    return abs(cv2.contourArea(np.asarray(points, dtype=np.float32)))


def resolve_landmarks(map_objects, names):
    by_name = defaultdict(list)
    for obj in map_objects.values():
        if obj.get("name"):
            by_name[normalize_name(obj["name"])].append(obj)
    resolved = []
    missing = []
    for name in names:
        matches = by_name.get(normalize_name(name), [])
        if matches:
            resolved.append(max(matches, key=lambda obj: polygon_area(obj["contour"])))
        else:
            missing.append(name)
    return resolved, missing


def choose_square_roi(target, landmarks, min_side_m, padding_m, full_landmark_max_span_m):
    points = list(target["contour"])
    landmark_modes = []
    target_polygon = Polygon(target["contour"])
    for landmark in landmarks:
        landmark_xy = np.asarray(landmark["contour"], dtype=np.float64)
        span = float(np.ptp(landmark_xy, axis=0).max())
        if span <= full_landmark_max_span_m:
            points.extend(landmark["contour"])
            landmark_modes.append("full_contour")
        else:
            _, local_landmark_point = nearest_points(target_polygon, Polygon(landmark["contour"]))
            points.append([local_landmark_point.x, local_landmark_point.y])
            landmark_modes.append("nearest_local_segment")
    xy = np.asarray(points, dtype=np.float64)
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    side = max(float(min_side_m), float(max(x_max - x_min, y_max - y_min) + 2 * padding_m))
    cx = float((x_min + x_max) / 2)
    cy = float((y_min + y_max) / 2)
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), side, landmark_modes


def extract_rgb(rgb, transform, roi, out_size):
    left, bottom, right, top = roi
    inv = ~transform
    src_left, src_top = inv * (left, top)
    src_right, src_bottom = inv * (right, bottom)
    x0, y0 = math.floor(src_left), math.floor(src_top)
    x1, y1 = math.ceil(src_right), math.ceil(src_bottom)
    canvas = np.zeros((max(1, y1 - y0), max(1, x1 - x0), 3), dtype=np.uint8)
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
    if ix1 > ix0 and iy1 > iy0:
        canvas[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = rgb[iy0:iy1, ix0:ix1]
    return cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)


def rasterize_target(contour, roi, out_size):
    left, bottom, right, top = roi
    side = right - left
    pts = np.asarray([
        [(x - left) / side * out_size, (top - y) / side * out_size]
        for x, y in contour
    ], dtype=np.float32)
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, out_size - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, out_size - 1)
    target_mask = np.zeros((out_size, out_size), dtype=np.uint8)
    cv2.fillPoly(target_mask, [pts], 1)
    return target_mask


def encode_rle(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=("train_seen", "val_seen", "val_unseen", "test_unseen"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--out-size", type=int, default=512)
    parser.add_argument("--min-side-m", type=float, default=160.0)
    parser.add_argument("--padding-m", type=float, default=20.0)
    parser.add_argument("--full-landmark-max-span-m", type=float, default=80.0)
    parser.add_argument("--text-model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path, default=Path("/home/tenant2/dataext/rsrefseg2/hf_cache"))
    parser.add_argument("--max-text-tokens", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    repo = args.repo.resolve()
    objects = json.loads((repo / "data/cityrefer/objects.json").read_text())
    processed = json.loads((repo / "data/cityrefer/processed_descriptions.json").read_text())
    rows = json.loads((repo / f"data/processed_citynav/citynav_{args.split}.json").read_text())
    if args.sample_count is not None:
        rng = random.Random(args.sample_seed)
        by_map = defaultdict(list)
        for row in rows:
            by_map[f"{row['area']}_block_{row['block']}"].append(row)
        for values in by_map.values():
            rng.shuffle(values)
        sampled = []
        map_names = sorted(by_map)
        while len(sampled) < min(args.sample_count, len(rows)):
            added = False
            for map_name in map_names:
                if by_map[map_name] and len(sampled) < args.sample_count:
                    sampled.append(by_map[map_name].pop())
                    added = True
            if not added:
                break
        rows = sampled
    if args.limit is not None:
        rows = rows[:args.limit]

    out = args.output_dir.resolve()
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out / f"cityrefer_{args.split}.jsonl"
    manifest_path = out / f"cityrefer_{args.split}_manifest.jsonl"

    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        map_name = f"{row['area']}_block_{row['block']}"
        grouped[map_name].append((index, row))

    annotations = [None] * len(rows)
    manifests = [None] * len(rows)
    tokenizer = AutoProcessor.from_pretrained(
        args.text_model, cache_dir=str(args.hf_cache), local_files_only=True
    ).tokenizer
    truncated_count = 0
    for map_name, indexed_rows in sorted(grouped.items()):
        rgb_path = repo / "data/rgbd" / f"{map_name}.png"
        tif_path = repo / "data/rgbd" / f"{map_name}.tif"
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        with rasterio.open(tif_path) as raster:
            transform = raster.transform
        for index, row in indexed_rows:
            object_id = str(row["object_ids"][0])
            ann_id = int(row["ann_ids"][0])
            target = objects[map_name][object_id]
            proc = processed[map_name][object_id][ann_id]
            landmarks, missing = resolve_landmarks(objects[map_name], proc.get("landmarks", []))
            roi, side_m, landmark_modes = choose_square_roi(
                target, landmarks, args.min_side_m, args.padding_m,
                args.full_landmark_max_span_m,
            )
            crop = extract_rgb(rgb, transform, roi, args.out_size)
            target_mask = rasterize_target(target["contour"], roi, args.out_size)
            file_name = f"{index:05d}_{map_name}_o{object_id}_a{ann_id}.jpg"
            if not cv2.imwrite(str(image_dir / file_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Failed to write {file_name}")

            instruction = row["descriptions"][0]
            original_ids = tokenizer(instruction, add_special_tokens=True)["input_ids"]
            encoded = tokenizer(
                instruction, add_special_tokens=True, truncation=True,
                max_length=args.max_text_tokens,
            )["input_ids"]
            model_instruction = tokenizer.decode(encoded, skip_special_tokens=True).strip()
            model_token_count = len(tokenizer(model_instruction, add_special_tokens=True)["input_ids"])
            if model_token_count > args.max_text_tokens:
                raise RuntimeError(f"Decoded text still has {model_token_count} tokens: {instruction}")
            was_truncated = len(original_ids) > args.max_text_tokens
            truncated_count += int(was_truncated)
            category = (
                f"lm{len(landmarks)}_"
                + ("context" if proc.get("surroundings") else "direct")
                + ("_missinglm" if missing else "")
            )
            annotations[index] = {
                "split": args.split,
                "image_id": str(index),
                "sent": model_instruction,
                "file_name": file_name,
                "segmentation": encode_rle(target_mask),
                "category_name": category,
            }
            manifests[index] = {
                "index": index,
                "map_name": map_name,
                "object_id": int(object_id),
                "ann_id": ann_id,
                "instruction": instruction,
                "model_instruction": model_instruction,
                "original_token_count": len(original_ids),
                "model_token_count": model_token_count,
                "text_truncated": was_truncated,
                "processed": proc,
                "resolved_landmarks": [obj["name"] for obj in landmarks],
                "landmark_roi_modes": landmark_modes,
                "missing_landmarks": missing,
                "roi_world": list(roi),
                "roi_side_m": side_m,
                "target_contour_world": target["contour"],
                "target_type": target["object_type"],
                "file_name": file_name,
                "target_mask_pixels": int(target_mask.sum()),
            }

    with ann_path.open("w") as stream:
        for item in annotations:
            stream.write(json.dumps(item) + "\n")
    with manifest_path.open("w") as stream:
        for item in manifests:
            stream.write(json.dumps(item) + "\n")
    summary = {
        "split": args.split,
        "samples": len(rows),
        "sample_seed": args.sample_seed if args.sample_count is not None else None,
        "maps": sorted(grouped),
        "out_size": args.out_size,
        "min_side_m": args.min_side_m,
        "padding_m": args.padding_m,
        "full_landmark_max_span_m": args.full_landmark_max_span_m,
        "max_text_tokens": args.max_text_tokens,
        "truncated_text_samples": truncated_count,
        "unresolved_landmark_samples": sum(bool(item["missing_landmarks"]) for item in manifests),
        "annotation_file": str(ann_path),
        "manifest_file": str(manifest_path),
    }
    (out / f"cityrefer_{args.split}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
