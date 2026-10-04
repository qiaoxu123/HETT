#!/usr/bin/env python3
"""Build object-grouped, target-centred orthophoto views and Phase-1 features."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiagent.cityreferobject import get_city_refer_objects  # noqa: E402
from multiagent.mapdata import GROUND_LEVEL  # noqa: E402
from multiagent.models.attribute_baseline import handcrafted_features  # noqa: E402
from multiagent.observation import cropclient  # noqa: E402
from multiagent.space import Pose4D  # noqa: E402
from multiagent.visual_attributes.dataset import leakage_report  # noqa: E402
from multiagent.visual_attributes.labels import (  # noqa: E402
    COLOR_CLASSES, SEMANTIC_CLASSES, SHAPE_CLASSES, SIZE_CLASSES, object_labels, size_thresholds,
)


SPLITS = ("train_seen", "val_seen", "val_unseen")


def project_contour(obj, center_x, center_y, altitude, image_size):
    result = []
    for point in obj.contour:
        row = (center_x + altitude - point.x) / (2 * altitude) * (image_size - 1)
        col = (center_y + altitude - point.y) / (2 * altitude) * (image_size - 1)
        result.append((int(round(col)), int(round(row))))
    return np.asarray(result, np.int32)


def representations(rgb, polygon, size):
    mask = cv2.fillPoly(np.zeros(rgb.shape[:2], np.uint8), [polygon], 1)
    masked = np.full_like(rgb, 127); masked[mask.astype(bool)] = rgb[mask.astype(bool)]
    x, y, width, height = cv2.boundingRect(polygon); padding = max(4, round(max(width, height) * 0.25))
    x0, y0 = max(0, x - padding), max(0, y - padding); x1, y1 = min(size, x + width + padding), min(size, y + height + padding)
    crop = rgb[y0:y1, x0:x1]
    if not crop.size: crop = rgb
    return {"whole": rgb, "crop": cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), "masked": masked}, mask


def collect_objects(data_root):
    objects = get_city_refer_objects(data_root / "cityrefer/objects.json", data_root / "cityrefer/processed_descriptions.json")
    grouped = {}
    for split in SPLITS:
        rows = json.loads((data_root / "processed_citynav" / f"citynav_{split}.json").read_text())
        by_object = defaultdict(lambda: {"texts": [], "episodes": []})
        for index, row in enumerate(rows):
            map_name = f"{row['area']}_block_{row['block']}"; object_id = int(row["object_ids"][0]); desc_id = int(row["ann_ids"][0])
            key = (map_name, object_id); by_object[key]["texts"].append(objects[map_name][object_id].descriptions[desc_id]); by_object[key]["episodes"].append(f"{split}:{index}")
        grouped[split] = by_object
    return objects, grouped


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--heights", nargs="+", type=int, default=(20, 40, 60, 80))
    parser.add_argument("--distances", nargs="+", type=int, default=(0,), help="target distance ahead of the camera")
    parser.add_argument("--image-size", type=int, default=256); parser.add_argument("--max-objects-per-split", type=int)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True); (args.output / "images").mkdir(exist_ok=True)
    objects, grouped = collect_objects(args.data_root)
    train_objects = [objects[map_name][object_id] for map_name, object_id in grouped["train_seen"]]
    thresholds = size_thresholds(train_objects); cropclient.load_image_cache(args.data_root / "rgbd")
    manifest, features = [], defaultdict(list)
    for split in SPLITS:
        entries = list(grouped[split].items())
        if args.max_objects_per_split: entries = entries[:args.max_objects_per_split]
        for item_index, ((map_name, object_id), metadata) in enumerate(entries):
            obj = objects[map_name][object_id]; labels = object_labels(obj, metadata["texts"], thresholds)
            object_key = f"{map_name}:{object_id}"
            for altitude in args.heights:
              for distance in args.distances:
                # Keep the entire target away from the exact image boundary.
                if distance > 0.75 * altitude: continue
                camera_x, camera_y = obj.position.x - distance, obj.position.y
                pose = Pose4D(camera_x, camera_y, GROUND_LEVEL[map_name] + altitude, 0.0)
                rgb = cropclient.crop_image(map_name, pose, (args.image_size, args.image_size), "rgb")
                polygon = project_contour(obj, camera_x, camera_y, altitude, args.image_size)
                images, mask = representations(rgb, polygon, args.image_size)
                sample_id = hashlib.sha1(f"{object_key}:{altitude}:{distance}:topdown".encode()).hexdigest()[:16]
                relative = Path("images") / f"{sample_id}.jpg"
                cv2.imwrite(str(args.output / relative), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                row = {"sample_id": sample_id, "split": split, "object_key": object_key,
                       "episode_ids": metadata["episodes"], "map_name": map_name,
                       "image": str(relative), "view": "topdown" if distance == 0 else "topdown_offset",
                       "height_m": altitude, "distance_m": float(distance), "labels": labels,
                       "mask_area_fraction": float(mask.mean()), "contour_uv": polygon.tolist()}
                manifest.append(row)
                for name, image in images.items(): features[name].append(handcrafted_features(image))
            if item_index and item_index % 500 == 0: print(f"{split}: {item_index}/{len(entries)}", flush=True)
    with (args.output / "manifest.jsonl").open("w") as stream:
        for row in manifest: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {"rows": manifest, "features": {name: torch.from_numpy(np.stack(value)) for name, value in features.items()},
               "classes": {"color": COLOR_CLASSES, "size": SIZE_CLASSES, "shape": SHAPE_CLASSES,
                           "semantic": SEMANTIC_CLASSES, "road_context": ("no", "yes"),
                           "roof_presence": ("no", "yes")},
               "size_thresholds_m2": thresholds}
    torch.save(payload, args.output / "handcrafted_features.pt")
    stats = {"objects": {split: len(grouped[split]) for split in SPLITS}, "samples": len(manifest),
             "heights": args.heights, "distances": args.distances,
             "size_thresholds_m2": thresholds, "leakage": leakage_report(manifest),
             "label_counts": {label: {split: defaultdict(int) for split in SPLITS} for label in payload["classes"]}}
    for row in manifest:
        for label in payload["classes"]:
            value = row["labels"].get(label)
            if value is not None: stats["label_counts"][label][row["split"]][value] += 1
    stats["label_counts"] = {label: {split: dict(values) for split, values in splits.items()} for label, splits in stats["label_counts"].items()}
    (args.output / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__": main()
