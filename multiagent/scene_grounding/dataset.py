"""Helpers for real teacher-pose scene-grounding samples."""
from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np
from shapely.geometry import Polygon

from multiagent.mapdata import GROUND_LEVEL
from multiagent.space import Pose4D, Pose5D, view_area_corners
from .metrics import DISTANCE_BUCKETS, distance_bucket


ALLOWED_SPLITS = ("train_seen", "val_seen", "val_unseen")


def reject_forbidden_split(split: str):
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split {split!r} is forbidden; allowed: {ALLOWED_SPLITS}")


def trajectory_pose(value) -> Pose4D:
    if len(value) == 6:
        return Pose5D.from_direction_vector(*map(float, value)).xyzyaw
    if len(value) == 4:
        return Pose4D(*map(float, value))
    raise ValueError(f"unsupported trajectory pose with {len(value)} fields")


def select_distance_steps(trajectory, goal_xy):
    """Select at most one real step per requested distance bucket."""
    goal = np.asarray(goal_xy, dtype=float)
    distances = np.linalg.norm(np.asarray([pose[:2] for pose in trajectory], dtype=float) - goal, axis=1)
    selected = []
    for lower, upper, name in DISTANCE_BUCKETS:
        indices = np.flatnonzero((distances >= lower) & (distances < upper))
        if not len(indices):
            continue
        target = (lower + upper) / 2 if np.isfinite(upper) else max(100.0, float(np.median(distances[indices])))
        index = int(indices[np.argmin(np.abs(distances[indices] - target))])
        selected.append((index, float(distances[index]), name))
    return selected


def world_to_image(points, pose: Pose4D, ground_level: float, image_size: int):
    altitude = pose.z - ground_level
    cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
    front, left = np.asarray([cosine, sine]), np.asarray([-sine, cosine])
    delta = np.asarray(points, dtype=float) - np.asarray([pose.x, pose.y])
    columns = (0.5 - delta @ left / (2 * altitude)) * (image_size - 1)
    rows = (0.5 - delta @ front / (2 * altitude)) * (image_size - 1)
    return np.stack((columns, rows), axis=-1)


def visible_regions(map_name, pose: Pose4D, objects, image_size=256, top_n=5):
    ground = GROUND_LEVEL[map_name]
    if pose.z <= ground:
        return []
    field = Polygon(view_area_corners(pose, ground))
    proposals = []
    for obj in objects:
        polygon = obj.contour_polygon
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        clipped = polygon.intersection(field)
        if clipped.is_empty or clipped.area < 1.0:
            continue
        geometry = max(getattr(clipped, "geoms", [clipped]), key=lambda item: item.area)
        if not hasattr(geometry, "exterior"):
            continue
        uv = world_to_image(np.asarray(geometry.exterior.coords), pose, ground, image_size)
        x, y, width, height = cv2.boundingRect(np.rint(uv).astype(np.int32))
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(image_size, x + width), min(image_size, y + height)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        proposals.append({"object_id": int(obj.id), "object_type": obj.object_type,
                          "bbox_xyxy": [x0, y0, x1, y1], "visible_area_m2": float(clipped.area),
                          "visible_fraction": float(clipped.area / max(polygon.area, 1e-6))})
    return sorted(proposals, key=lambda item: item["visible_area_m2"], reverse=True)[:top_n]


def sample_id(split, episode_index, step):
    return hashlib.sha1(f"{split}:{episode_index}:{step}:teacher-pose".encode()).hexdigest()[:16]


def leakage_audit(rows):
    by_split = {}
    for split in ALLOWED_SPLITS:
        selected = [row for row in rows if row["split"] == split]
        by_split[split] = {
            "episodes": {row["episode_id"] for row in selected},
            "objects": {row["object_key"] for row in selected},
            "samples": {row["sample_id"] for row in selected},
        }
    overlap = {}
    for index, left in enumerate(ALLOWED_SPLITS):
        for right in ALLOWED_SPLITS[index + 1:]:
            overlap[f"{left}/{right}"] = {
                key: len(by_split[left][key] & by_split[right][key])
                for key in ("episodes", "objects", "samples")
            }
    return {"overlap": overlap, "forbidden_test_unseen": any(row.get("split") == "test_unseen" for row in rows),
            "candidate_centered_observations": any(row.get("observation_source") != "teacher_trajectory_pose" for row in rows)}
