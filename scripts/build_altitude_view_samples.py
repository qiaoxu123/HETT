#!/usr/bin/env python3
"""Build a small visual audit of altitude-dependent RGB observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio

from multiagent.dataset.mturk_trajectory import MTurkTrajectory
from multiagent.mapdata import GROUND_LEVEL
from multiagent.space import Pose4D, view_area_corners


SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
VIEW_MIN_METERS = 20.0
VIEW_MAX_METERS = 120.0


def crop_view(image, raster, pose, ground, size):
    corners = np.array(
        [raster.index(point.x, point.y) for point in view_area_corners(pose, ground)],
        dtype=np.float32,
    )
    source = np.flip(corners, axis=-1)
    target = np.array(
        [(0, 0), (size - 1, 0), (size - 1, size - 1), (0, size - 1)],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, transform, (size, size), flags=cv2.INTER_LINEAR)


def save_jpeg(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 91])


def select_trajectories(annotation_root, rgbd_root):
    selected = []
    for split in ("val_seen", "val_unseen"):
        rows = json.loads((annotation_root / f"citynav_{split}.json").read_text())
        candidates = []
        for raw in rows:
            trajectory = MTurkTrajectory(**raw)
            map_name = trajectory.map_name
            if not (rgbd_root / f"{map_name}.png").exists() or len(trajectory.trajectory) < 25:
                continue
            ground = GROUND_LEVEL[map_name]
            heights = np.array([pose.z - ground for pose in trajectory.trajectory])
            descent = heights[0] - heights[-1]
            description = trajectory.descriptions[0]
            if heights[0] < 55 or heights[-1] > 45 or descent < 35:
                continue
            score = descent + min(len(description.split()), 35)
            candidates.append((score, trajectory, heights))

        used_maps = set()
        for _, trajectory, heights in sorted(candidates, key=lambda item: item[0], reverse=True):
            if trajectory.map_name in used_maps:
                continue
            selected.append((split, trajectory, heights))
            used_maps.add(trajectory.map_name)
            if len(used_maps) == 2:
                break
    if len(selected) != 4:
        raise RuntimeError(f"Expected four representative trajectories, found {len(selected)}")
    return selected


def build_overview(image, raster, trajectory, ground, indices, output_path):
    canvas = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    points = []
    for pose in trajectory.trajectory:
        row, col = raster.index(pose.x, pose.y)
        points.append((int(col), int(row)))
    cv2.polylines(canvas, [np.array(points, np.int32)], False, (40, 220, 255), 8, cv2.LINE_AA)

    colors = [(255, 90, 60), (255, 155, 50), (70, 210, 120), (90, 170, 255), (210, 100, 255)]
    for number, (index, color) in enumerate(zip(indices, colors), start=1):
        pose5 = trajectory.trajectory[index]
        view_height = float(np.clip(pose5.z - ground, VIEW_MIN_METERS, VIEW_MAX_METERS))
        pose = Pose4D(pose5.x, pose5.y, ground + view_height, pose5.yaw)
        footprint = np.array(
            [np.flip(raster.index(p.x, p.y)) for p in view_area_corners(pose, ground)],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [footprint], True, color, 5, cv2.LINE_AA)
        center = tuple(np.flip(raster.index(pose.x, pose.y)))
        center = (int(center[0]), int(center[1]))
        cv2.circle(canvas, center, 15, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(number), (center[0] + 18, center[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    target = trajectory.target_position
    target_rc = raster.index(target.x, target.y)
    target_xy = (int(target_rc[1]), int(target_rc[0]))
    cv2.drawMarker(canvas, target_xy, (20, 20, 245), cv2.MARKER_TILTED_CROSS, 48, 8, cv2.LINE_AA)

    scale = min(1.0, 1100 / canvas.shape[1])
    overview = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overview, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("data/processed_citynav"))
    parser.add_argument("--rgbd-root", type=Path, default=Path("data/rgbd"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assets = args.output / "assets"
    records = []
    for trajectory_id, (split, trajectory, heights) in enumerate(
        select_trajectories(args.annotation_root, args.rgbd_root), start=1
    ):
        map_name = trajectory.map_name
        image = cv2.cvtColor(cv2.imread(str(args.rgbd_root / f"{map_name}.png")), cv2.COLOR_BGR2RGB)
        raster = rasterio.open(args.rgbd_root / f"{map_name}.tif")
        ground = GROUND_LEVEL[map_name]
        indices = sorted({round(fraction * (len(trajectory.trajectory) - 1)) for fraction in SAMPLE_FRACTIONS})
        prefix = f"trajectory-{trajectory_id}"
        build_overview(image, raster, trajectory, ground, indices, assets / f"{prefix}-overview.jpg")

        steps = []
        target = trajectory.target_position
        for sample_number, index in enumerate(indices, start=1):
            source_pose = trajectory.trajectory[index]
            raw_height = float(source_pose.z - ground)
            view_height = float(np.clip(raw_height, VIEW_MIN_METERS, VIEW_MAX_METERS))
            dynamic_pose = Pose4D(source_pose.x, source_pose.y, ground + view_height, source_pose.yaw)
            fixed_pose = Pose4D(source_pose.x, source_pose.y, ground + 50.0, source_pose.yaw)

            fixed = crop_view(image, raster, fixed_pose, ground, 224)
            dynamic = crop_view(image, raster, dynamic_pose, ground, 224)
            highres = crop_view(image, raster, dynamic_pose, ground, 448)
            compressed = cv2.resize(highres, (224, 224), interpolation=cv2.INTER_AREA)

            base = assets / f"{prefix}-step-{sample_number}"
            save_jpeg(base.with_name(base.name + "-fixed.jpg"), fixed)
            save_jpeg(base.with_name(base.name + "-dynamic.jpg"), dynamic)
            save_jpeg(base.with_name(base.name + "-compressed.jpg"), compressed)

            distance = float(np.hypot(source_pose.x - target.x, source_pose.y - target.y))
            steps.append(
                {
                    "number": sample_number,
                    "trajectory_index": index,
                    "fraction": index / max(1, len(trajectory.trajectory) - 1),
                    "raw_height": round(raw_height, 1),
                    "view_height": round(view_height, 1),
                    "distance_to_target": round(distance, 1),
                    "dynamic_fov": round(2 * view_height, 1),
                    "fixed": f"assets/{prefix}-step-{sample_number}-fixed.jpg",
                    "dynamic": f"assets/{prefix}-step-{sample_number}-dynamic.jpg",
                    "compressed": f"assets/{prefix}-step-{sample_number}-compressed.jpg",
                }
            )

        records.append(
            {
                "id": trajectory_id,
                "split": split,
                "map_name": map_name,
                "description": trajectory.descriptions[0],
                "trajectory_points": len(trajectory.trajectory),
                "overview": f"assets/{prefix}-overview.jpg",
                "height_profile": [round(float(value), 1) for value in heights],
                "steps": steps,
            }
        )
        raster.close()

    (args.output / "data.json").write_text(json.dumps({"trajectories": records}, ensure_ascii=False, indent=2))
    print(f"Wrote {len(records)} trajectories and {sum(len(r['steps']) for r in records)} sampled steps")


if __name__ == "__main__":
    main()
