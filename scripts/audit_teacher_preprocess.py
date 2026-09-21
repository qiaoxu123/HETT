#!/usr/bin/env python3
"""Offline audit of landmark-aware human trajectory cleanup."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiagent.cityreferobject import get_city_refer_objects  # noqa: E402
from multiagent.dataset.generate import _description_landmark_contours  # noqa: E402
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories  # noqa: E402
from multiagent.teacher.preprocess import optimize_teacher_path  # noqa: E402


def length(points):
    xyz = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/teacher_preprocess_audit")
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--coarse-moves", type=int, default=10)
    parser.add_argument("--local-moves", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    objects = get_city_refer_objects()
    result = {}
    rows = []
    for split in ("train_seen", "val_seen", "val_unseen"):
        trajectories = [item for item in load_mturk_trajectories(split, "all", 50)
                        if item.dist_marker_to_target <= 30]
        part = []
        for item in trajectories:
            source = item.interpolated_pose_trajectory
            optimized = optimize_teacher_path(
                source, _description_landmark_contours(objects, item),
                arrival_radius=args.radius,
                coarse_moves=args.coarse_moves,
                local_moves=args.local_moves,
            )
            row = {"split": split, "id": [item.map_name, item.object_id, item.desc_id],
                   "optimized": optimized is not None, "source_points": len(source)}
            if optimized is not None:
                clean = [pose.xyz for pose in optimized.poses]
                original_length = length(source)
                clean_length = length(clean)
                row.update({
                    "cleaned_points": len(clean), "stage_boundary": optimized.stage_boundary,
                    "coarse_moves": optimized.stage_boundary,
                    "local_moves": len(clean) - optimized.stage_boundary - 1,
                    "source_length_m": original_length, "cleaned_length_m": clean_length,
                    "length_ratio": clean_length / original_length if original_length else 1.0,
                    "endpoint_error_m": float(np.linalg.norm(np.asarray(clean[-1]) - np.asarray(source[-1].xyz))),
                })
            part.append(row)
        good = [row for row in part if row["optimized"]]
        result[split] = {
            "episodes": len(part), "optimized": len(good),
            "optimized_percent": 100 * len(good) / len(part),
            "source_points": describe([row["source_points"] for row in good]),
            "cleaned_points": describe([row["cleaned_points"] for row in good]),
            "coarse_moves": describe([row["coarse_moves"] for row in good]),
            "local_moves": describe([row["local_moves"] for row in good]),
            "length_ratio": describe([row["length_ratio"] for row in good]),
            "maximum_endpoint_error_m": max(row["endpoint_error_m"] for row in good),
        }
        rows.extend(part)
    output = {"parameters": vars(args) | {"out": str(args.out)}, "summary": result, "rows": rows}
    (args.out / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
