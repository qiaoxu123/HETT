"""Turn a human route into explicit coarse-arrival and local-exploration phases."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from multiagent.space import Point2D, Point3D, Pose4D


@dataclass(frozen=True)
class OptimizedTeacherPath:
    poses: List[Pose4D]
    stage_boundary: int
    source_arrival_index: int
    source_points: int
    cleaned_points: int


def _array(points):
    return np.asarray([[float(p.x), float(p.y), float(p.z), float(getattr(p, "yaw", np.nan))]
                       for p in points], dtype=np.float64)


def _distance_to_contours(xy, contours):
    nearest = float("inf")
    point = tuple(float(value) for value in xy)
    for contour in contours:
        polygon = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(polygon) >= 3:
            nearest = min(nearest, max(0.0, -float(cv2.pointPolygonTest(polygon, point, True))))
    return nearest


def find_stable_arrival(points, contours, radius=20.0, stable_points=2) -> Optional[int]:
    """First point for which this and the next points remain inside the radius."""
    if not contours or len(points) < stable_points:
        return None
    inside = np.asarray([
        _distance_to_contours((point.x, point.y), contours) <= radius for point in points
    ])
    for index in range(len(points) - stable_points + 1):
        if inside[index:index + stable_points].all():
            return index
    return None


def _deduplicate(points, minimum_spacing=1.0):
    kept = [points[0]]
    for point in points[1:-1]:
        if np.linalg.norm(point[:2] - kept[-1][:2]) >= minimum_spacing:
            kept.append(point)
    if len(points) > 1 and np.linalg.norm(points[-1, :3] - kept[-1][:3]) > 1e-6:
        kept.append(points[-1])
    return np.asarray(kept)


def _remove_loops(points, loop_radius=7.5):
    """Remove a detour when the route returns near an earlier location."""
    kept = []
    for point in points:
        match = None
        for index in range(len(kept) - 2, -1, -1):
            if np.linalg.norm(point[:2] - kept[index][:2]) <= loop_radius:
                match = index
                break
        if match is not None:
            kept = kept[:match + 1]
            kept[-1] = point
        elif not kept or np.linalg.norm(point[:3] - kept[-1][:3]) > 1e-6:
            kept.append(point)
    return np.asarray(kept)


def _rdp(points, epsilon=4.0):
    if len(points) <= 2:
        return points
    start, end = points[0, :2], points[-1, :2]
    segment = end - start
    norm = np.linalg.norm(segment)
    if norm < 1e-6:
        distances = np.linalg.norm(points[1:-1, :2] - start, axis=1)
    else:
        offsets = points[1:-1, :2] - start
        distances = np.abs(segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0]) / norm
    split = int(np.argmax(distances)) + 1
    if distances[split - 1] <= epsilon:
        return np.stack([points[0], points[-1]])
    return np.concatenate([_rdp(points[:split + 1], epsilon)[:-1], _rdp(points[split:], epsilon)])


def _limit_by_arclength(points, maximum_points):
    if len(points) <= maximum_points:
        return points
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))])
    targets = np.linspace(0, cumulative[-1], maximum_points)
    indices = [int(np.argmin(np.abs(cumulative - target))) for target in targets]
    indices[0], indices[-1] = 0, len(points) - 1
    return points[np.unique(indices)]


def _clean_phase(points, maximum_moves, preserve_observations=False):
    original_start, original_end = points[0].copy(), points[-1].copy()
    points = _deduplicate(points)
    points = _remove_loops(points)
    if len(points) == 1:
        points = np.stack([original_start, original_end])
    else:
        points[0], points[-1] = original_start, original_end
    if not preserve_observations:
        points = _rdp(points)
    return _limit_by_arclength(points, maximum_moves + 1)


def _to_poses(points):
    poses = []
    previous_yaw = 0.0
    for index, point in enumerate(points):
        supplied_yaw = point[3] if len(point) > 3 else np.nan
        if np.isfinite(supplied_yaw):
            previous_yaw = float(supplied_yaw)
        elif index + 1 < len(points):
            delta = points[index + 1, :2] - point[:2]
            if np.linalg.norm(delta) > 1e-6:
                previous_yaw = float(np.arctan2(delta[1], delta[0]))
        poses.append(Pose4D(float(point[0]), float(point[1]), float(point[2]), previous_yaw))
    return poses


def optimize_teacher_path(
    trajectory: Sequence[Point3D],
    landmark_contours: Sequence[Sequence[Point2D]],
    arrival_radius=20.0,
    stable_points=2,
    coarse_moves=10,
    local_moves=10,
) -> Optional[OptimizedTeacherPath]:
    """Clean and budget a human route while keeping an explicit phase boundary.

    Returns ``None`` when the human route never stably reaches a supplied
    landmark. Callers should preserve the original route for those examples.
    """
    arrival = find_stable_arrival(trajectory, landmark_contours, arrival_radius, stable_points)
    if arrival is None or arrival == 0 or arrival >= len(trajectory) - 1:
        return None
    points = _array(trajectory)
    coarse = _clean_phase(points[:arrival + 1], coarse_moves)
    # Straight local movement still provides new visual evidence, so it must not
    # collapse to two endpoints as coarse navigation may.
    local = _clean_phase(points[arrival:], local_moves, preserve_observations=True)
    combined = np.concatenate([coarse, local[1:]])
    boundary = len(coarse) - 1
    return OptimizedTeacherPath(
        poses=_to_poses(combined),
        stage_boundary=boundary,
        source_arrival_index=arrival,
        source_points=len(points),
        cleaned_points=len(combined),
    )
