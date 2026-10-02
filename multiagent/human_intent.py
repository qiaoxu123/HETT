"""Human-demonstration intent reconstruction for CityFlight trajectories.

The raw CityFlight teacher trace is a first-person keyboard-control trace.  It
contains useful route/observation intent together with small control jitter.
This module keeps the raw data untouched and derives a compact intent path plus
fixed-distance future targets for supervision.
"""

from __future__ import annotations

from math import pi
from typing import Iterable, Sequence

import numpy as np

from multiagent.space import Pose4D


def _wrap_angle(theta: float) -> float:
    return (theta + pi) % (2 * pi) - pi


def _angle_distance(a: float, b: float) -> float:
    return abs(_wrap_angle(a - b))


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom < 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, delta) / denom)
    ratio = min(1.0, max(0.0, ratio))
    projection = start + ratio * delta
    return float(np.linalg.norm(point - projection))


def _rdp_indices(points: np.ndarray, tolerance: float) -> list[int]:
    """Return Ramer-Douglas-Peucker indices while preserving point order."""
    n_points = len(points)
    if n_points <= 2 or tolerance <= 0:
        return list(range(n_points))

    keep = {0, n_points - 1}
    stack = [(0, n_points - 1)]
    while stack:
        start_idx, end_idx = stack.pop()
        if end_idx <= start_idx + 1:
            continue

        start = points[start_idx]
        end = points[end_idx]
        max_distance = -1.0
        max_idx = None
        for idx in range(start_idx + 1, end_idx):
            distance = _point_segment_distance(points[idx], start, end)
            if distance > max_distance:
                max_distance = distance
                max_idx = idx

        if max_idx is not None and max_distance > tolerance:
            keep.add(max_idx)
            stack.append((start_idx, max_idx))
            stack.append((max_idx, end_idx))

    return sorted(keep)


def reconstruct_human_intent_path(
    trajectory: Sequence[Pose4D],
    min_step_m: float = 1.0,
    rdp_tolerance_m: float = 2.5,
    yaw_keyframe_deg: float = 30.0,
) -> list[Pose4D]:
    """Remove small keyboard jitter while preserving route and view changes.

    Spatial structure is simplified with RDP.  Large yaw changes are kept even
    when the UAV barely moves, because first-person observation changes are part
    of the human demonstration rather than mere XY path geometry.
    """
    if len(trajectory) <= 2:
        return list(trajectory)

    yaw_threshold = np.deg2rad(yaw_keyframe_deg)

    # First remove only consecutive points that are both spatially tiny and do
    # not change the viewing direction meaningfully.
    filtered = [trajectory[0]]
    filtered_original_indices = [0]
    for original_idx, pose in enumerate(trajectory[1:-1], start=1):
        previous = filtered[-1]
        step = np.linalg.norm(
            np.array([pose.x - previous.x, pose.y - previous.y], dtype=np.float32)
        )
        yaw_change = _angle_distance(pose.yaw, previous.yaw)
        if step >= min_step_m or yaw_change >= yaw_threshold:
            filtered.append(pose)
            filtered_original_indices.append(original_idx)

    filtered.append(trajectory[-1])
    filtered_original_indices.append(len(trajectory) - 1)

    if len(filtered) <= 2:
        return filtered

    xy = np.array([[pose.x, pose.y] for pose in filtered], dtype=np.float32)
    keep = set(_rdp_indices(xy, rdp_tolerance_m))

    # Preserve observation/view keyframes independently of XY simplification.
    last_yaw_idx = 0
    for idx in range(1, len(filtered) - 1):
        if _angle_distance(filtered[idx].yaw, filtered[last_yaw_idx].yaw) >= yaw_threshold:
            keep.add(idx)
            last_yaw_idx = idx

    keep.add(0)
    keep.add(len(filtered) - 1)
    return [filtered[idx] for idx in sorted(keep)]


def _polyline_cumulative_distance(path: Sequence[Pose4D]) -> np.ndarray:
    cumulative = np.zeros(len(path), dtype=np.float32)
    if len(path) <= 1:
        return cumulative
    segment_lengths = [
        np.hypot(path[idx + 1].x - path[idx].x, path[idx + 1].y - path[idx].y)
        for idx in range(len(path) - 1)
    ]
    cumulative[1:] = np.cumsum(segment_lengths, dtype=np.float32)
    return cumulative


def project_xy_to_intent_path(
    current_xy: Iterable[float],
    path: Sequence[Pose4D],
) -> tuple[float, float]:
    """Project current XY onto the intent polyline.

    Returns (arc_length_position, euclidean_projection_error).
    """
    if len(path) == 0:
        return 0.0, float("inf")
    if len(path) == 1:
        point = np.asarray(current_xy, dtype=np.float32)
        anchor = np.array([path[0].x, path[0].y], dtype=np.float32)
        return 0.0, float(np.linalg.norm(point - anchor))

    point = np.asarray(current_xy, dtype=np.float32)
    cumulative = _polyline_cumulative_distance(path)
    best_s = 0.0
    best_distance = float("inf")

    for idx in range(len(path) - 1):
        start = np.array([path[idx].x, path[idx].y], dtype=np.float32)
        end = np.array([path[idx + 1].x, path[idx + 1].y], dtype=np.float32)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-8:
            distance = float(np.linalg.norm(point - start))
            if distance < best_distance:
                best_distance = distance
                best_s = float(cumulative[idx])
            continue

        ratio = float(np.dot(point - start, delta) / (length * length))
        ratio = min(1.0, max(0.0, ratio))
        projection = start + ratio * delta
        distance = float(np.linalg.norm(point - projection))
        if distance < best_distance:
            best_distance = distance
            best_s = float(cumulative[idx] + ratio * length)

    return best_s, best_distance


def sample_intent_pose_at_distance(
    path: Sequence[Pose4D],
    distance_m: float,
) -> Pose4D:
    """Interpolate XY/Z/yaw at an arc-length coordinate on the intent path."""
    if len(path) == 0:
        raise ValueError("intent path must contain at least one pose")
    if len(path) == 1:
        return path[0]

    cumulative = _polyline_cumulative_distance(path)
    total = float(cumulative[-1])
    distance_m = min(total, max(0.0, float(distance_m)))

    if distance_m >= total - 1e-8:
        return path[-1]

    upper = int(np.searchsorted(cumulative, distance_m, side="right"))
    upper = min(max(upper, 1), len(path) - 1)
    lower = upper - 1

    # Repeated cumulative values correspond to pure view rotations.  Skip
    # forward to the next spatial segment so the latest yaw at that location is
    # used as the departure orientation.
    while upper < len(path) - 1 and cumulative[upper] <= cumulative[lower] + 1e-8:
        lower = upper
        upper += 1

    segment_length = float(cumulative[upper] - cumulative[lower])
    if segment_length < 1e-8:
        return path[upper]

    ratio = float((distance_m - cumulative[lower]) / segment_length)
    ratio = min(1.0, max(0.0, ratio))
    start = path[lower]
    end = path[upper]
    yaw_delta = _wrap_angle(end.yaw - start.yaw)

    return Pose4D(
        start.x + ratio * (end.x - start.x),
        start.y + ratio * (end.y - start.y),
        start.z + ratio * (end.z - start.z),
        _wrap_angle(start.yaw + ratio * yaw_delta),
    )


def build_fixed_horizon_intent_targets(
    intent_path: Sequence[Pose4D],
    current_xy: Iterable[float],
    horizons_m: Sequence[float],
    goal_xy: Iterable[float] | None = None,
) -> dict[str, np.ndarray | float]:
    """Build fixed-distance human-intent targets from the current state.

    Fixed horizons are valid only if that much intent path remains.  A final
    goal target is always appended and valid, avoiding repeated clamped goal
    copies when the UAV is already near the destination.
    """
    if len(intent_path) == 0:
        raise ValueError("intent path must contain at least one pose")

    cumulative = _polyline_cumulative_distance(intent_path)
    total = float(cumulative[-1])
    current_s, projection_error = project_xy_to_intent_path(current_xy, intent_path)
    remaining = max(0.0, total - current_s)

    poses = []
    valid = []
    for horizon in horizons_m:
        horizon = float(horizon)
        is_valid = horizon <= remaining + 1e-6
        sample_s = min(total, current_s + horizon)
        poses.append(sample_intent_pose_at_distance(intent_path, sample_s))
        valid.append(float(is_valid))

    final_pose = intent_path[-1]
    if goal_xy is None:
        goal_x, goal_y = final_pose.x, final_pose.y
    else:
        goal_x, goal_y = [float(value) for value in goal_xy]
    poses.append(Pose4D(goal_x, goal_y, final_pose.z, final_pose.yaw))
    valid.append(1.0)

    xy = np.asarray([[pose.x, pose.y] for pose in poses], dtype=np.float32)
    yaw = np.asarray([pose.yaw for pose in poses], dtype=np.float32)
    heading = np.stack((np.sin(yaw), np.cos(yaw)), axis=1).astype(np.float32)

    return {
        "xy": xy,
        "yaw": yaw,
        "heading": heading,
        "valid": np.asarray(valid, dtype=np.float32),
        "remaining_distance_m": np.float32(remaining),
        "projection_error_m": np.float32(projection_error),
    }
