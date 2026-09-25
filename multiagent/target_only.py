"""Utilities for target-only keyframe supervision.

The helpers in this file are deliberately independent from the simulator.  In
particular, ground-truth target coordinates are used only to select *training*
keyframes and to build labels/metrics.  They are never returned as model
inputs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def resample_contour(points: Sequence, count: int = 16) -> np.ndarray:
    """Uniformly sample a closed polygon boundary to ``count`` XY points."""
    xy = np.asarray([[point.x, point.y] if hasattr(point, "x") else point[:2]
                     for point in points], dtype=np.float32)
    if len(xy) == 0:
        return np.zeros((count, 2), dtype=np.float32)
    if len(xy) == 1:
        return np.repeat(xy, count, axis=0)
    closed = np.concatenate((xy, xy[:1]), axis=0)
    segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(segment.sum())
    if perimeter <= 1e-6:
        return np.repeat(xy[:1], count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    samples = np.linspace(0.0, perimeter, count, endpoint=False)
    result = np.empty((count, 2), dtype=np.float32)
    for index, distance in enumerate(samples):
        edge = min(int(np.searchsorted(cumulative, distance, side="right") - 1),
                   len(segment) - 1)
        fraction = (distance - cumulative[edge]) / max(float(segment[edge]), 1e-6)
        result[index] = closed[edge] + fraction * (closed[edge + 1] - closed[edge])
    return result


def pose_goal_distance(pose, target) -> float:
    """Horizontal metric distance without depending on a concrete pose class."""
    pose_xy = pose.xy if hasattr(pose, "xy") else pose
    target_xy = target.xy if hasattr(target, "xy") else target
    if hasattr(pose_xy, "dist_to"):
        return float(pose_xy.dist_to(target_xy))
    pose_array = np.asarray(pose_xy, dtype=np.float32)[..., :2]
    target_array = np.asarray(target_xy, dtype=np.float32)[..., :2]
    return float(np.linalg.norm(pose_array - target_array))


def select_keyframe_pose(
    trajectory: Sequence,
    target,
    mode: str,
    near_min_m: float = 25.0,
    near_max_m: float = 60.0,
    rng=None,
):
    """Select one informative frame while avoiding the 20 m success-circle leak.

    ``near_ring`` first selects a frame in the configured annulus.  If a human
    trajectory has no such frame, it chooses the closest frame that is still
    outside ``near_min_m``; only a trajectory wholly inside that radius falls
    back to its farthest frame.  ``closest`` is retained solely as a privileged
    upper-bound diagnostic, while ``start`` is the deployable initial-view
    diagnostic.
    """
    if not trajectory:
        raise ValueError("trajectory must contain at least one pose")
    distances = np.asarray([pose_goal_distance(pose, target) for pose in trajectory])
    if mode == "start":
        return trajectory[0]
    if mode == "closest":
        return trajectory[int(distances.argmin())]
    if mode != "near_ring":
        raise ValueError(f"unknown keyframe mode: {mode}")
    ring = np.flatnonzero((distances >= near_min_m) & (distances <= near_max_m))
    if len(ring):
        if rng is None:
            return trajectory[int(ring[len(ring) // 2])]
        return trajectory[int(rng.choice(ring))]
    outside = np.flatnonzero(distances >= near_min_m)
    if len(outside):
        return trajectory[int(outside[np.argmin(distances[outside])])]
    return trajectory[int(distances.argmax())]


def target_prediction_metrics(errors_m: Sequence[float]) -> dict:
    errors = np.asarray(errors_m, dtype=np.float64)
    if errors.size == 0:
        return {"count": 0, "mean_error_m": float("nan"),
                "median_error_m": float("nan"), "hit20": float("nan"),
                "hit50": float("nan")}
    return {
        "count": int(errors.size),
        "mean_error_m": float(errors.mean()),
        "median_error_m": float(np.median(errors)),
        "hit20": float((errors <= 20.0).mean() * 100.0),
        "hit50": float((errors <= 50.0).mean() * 100.0),
    }
