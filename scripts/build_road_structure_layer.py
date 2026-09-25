#!/usr/bin/env python3
"""Rasterize CityRefer road polygons into map-aligned binary structure layers."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiagent.mapdata import MAP_BOUNDS


def contour_to_pixels(contour, map_name, resolution=256, map_meters=410.0):
    bounds = MAP_BOUNDS[map_name]
    points = np.asarray(contour, dtype=np.float32)
    x = (points[:, 0] - bounds.x_min) / map_meters * (resolution - 1)
    y = (bounds.y_max - points[:, 1]) / map_meters * (resolution - 1)
    return np.rint(np.stack((x, y), axis=-1)).astype(np.int32)


def build_binary_layer(objects, map_name, resolution=256,
                       object_types=("TrafficRoad",)):
    """Return a uint8 {0,1} mask aligned to normalized CityNav coordinates."""
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    selected = 0
    for obj in objects[map_name].values():
        if obj.get("object_type") not in object_types:
            continue
        points = contour_to_pixels(obj["contour"], map_name, resolution)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
            selected += 1
    return mask, selected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/road_structure_layers"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--include-footpath", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    objects = json.load(open(args.data_root / "cityrefer/objects.json"))
    object_types = ("TrafficRoad", "Footpath") if args.include_footpath else ("TrafficRoad",)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for map_name in sorted(objects):
        mask, count = build_binary_layer(
            objects, map_name, args.resolution, object_types)
        cv2.imwrite(str(args.output_dir / f"{map_name}.png"), mask * 255)
        summary[map_name] = {
            "polygon_count": count,
            "coverage_percent": float(mask.mean() * 100.0),
        }
    with open(args.output_dir / "summary.json", "w") as stream:
        json.dump({"object_types": object_types, "resolution": args.resolution,
                   "maps": summary}, stream, indent=2)
    print(json.dumps({
        "maps": len(summary),
        "polygons": sum(row["polygon_count"] for row in summary.values()),
        "mean_coverage_percent": sum(row["coverage_percent"] for row in summary.values())
                                 / max(len(summary), 1),
    }, indent=2))


if __name__ == "__main__":
    main()
