"""Human-intent core anchors for trajectory-grounded aerial navigation.

The raw CityFlight trace contains dense keyboard-control samples.  This module
extracts sparse decision anchors from three cues that are available in the
human demonstration:

1. geometric route changes (RDP corners),
2. large first-person yaw/view changes,
3. closest passages to instruction-referenced landmarks.

The anchors are training supervision only.  At inference the policy predicts
the next core anchor from language, landmark semantics, current RGB evidence,
map/history context and the final-goal heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable, Sequence

import numpy as np

from multiagent.space import Pose4D


def _wrap_angle(theta: float) -> float:
    return (float(theta) + pi) % (2.0 * pi) - pi


def _angle_distance(a: float, b: float) -> float:
    return abs(_wrap_angle(float(a) - float(b)))


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom < 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, delta) / denom)
    ratio = min(1.0, max(0.0, ratio))
    projection = start + ratio * delta
    return float(np.linalg.norm(point - projection))


def _rdp_indices(points: np.ndarray, tolerance: float) -> list[int]:
    """Ramer-Douglas-Peucker indices while preserving path order."""
    if len(points) <= 2 or tolerance <= 0:
        return list(range(len(points)))

    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start_idx, end_idx = stack.pop()
        if end_idx <= start_idx + 1:
            continue

        max_distance = -1.0
        max_idx = None
        for idx in range(start_idx + 1, end_idx):
            distance = _point_segment_distance(
                points[idx],
                points[start_idx],
                points[end_idx],
            )
            if distance > max_distance:
                max_distance = distance
                max_idx = idx

        if max_idx is not None and max_distance > tolerance:
            keep.add(max_idx)
            stack.append((start_idx, max_idx))
            stack.append((max_idx, end_idx))

    return sorted(keep)


def _cumulative_xy_distance(trajectory: Sequence[Pose4D]) -> np.ndarray:
    cumulative = np.zeros(len(trajectory), dtype=np.float32)
    if len(trajectory) <= 1:
        return cumulative
    segment_lengths = [
        np.hypot(
            trajectory[idx + 1].x - trajectory[idx].x,
            trajectory[idx + 1].y - trajectory[idx].y,
        )
        for idx in range(len(trajectory) - 1)
    ]
    cumulative[1:] = np.cumsum(segment_lengths, dtype=np.float32)
    return cumulative


def _project_xy_to_path(
    current_xy: Iterable[float],
    trajectory: Sequence[Pose4D],
    cumulative: np.ndarray,
) -> tuple[float, float]:
    if len(trajectory) == 0:
        return 0.0, float("inf")
    if len(trajectory) == 1:
        point = np.asarray(current_xy, dtype=np.float32)
        anchor = np.array(
            [trajectory[0].x, trajectory[0].y],
            dtype=np.float32,
        )
        return 0.0, float(np.linalg.norm(point - anchor))

    point = np.asarray(current_xy, dtype=np.float32)
    best_arc = 0.0
    best_error = float("inf")

    for idx in range(len(trajectory) - 1):
        start = np.array(
            [trajectory[idx].x, trajectory[idx].y],
            dtype=np.float32,
        )
        end = np.array(
            [trajectory[idx + 1].x, trajectory[idx + 1].y],
            dtype=np.float32,
        )
        delta = end - start
        length_sq = float(np.dot(delta, delta))
        if length_sq < 1e-12:
            ratio = 0.0
            projection = start
        else:
            ratio = float(np.dot(point - start, delta) / length_sq)
            ratio = min(1.0, max(0.0, ratio))
            projection = start + ratio * delta

        error = float(np.linalg.norm(point - projection))
        if error < best_error:
            length = float(np.sqrt(max(length_sq, 0.0)))
            best_error = error
            best_arc = float(cumulative[idx] + ratio * length)

    return best_arc, best_error


def _sample_xy_at_arc(
    trajectory: Sequence[Pose4D],
    cumulative: np.ndarray,
    arc_m: float,
) -> np.ndarray:
    if len(trajectory) == 1:
        return np.array(
            [trajectory[0].x, trajectory[0].y],
            dtype=np.float32,
        )

    arc_m = float(np.clip(arc_m, 0.0, cumulative[-1]))
    if arc_m >= float(cumulative[-1]) - 1e-8:
        return np.array(
            [trajectory[-1].x, trajectory[-1].y],
            dtype=np.float32,
        )

    upper = int(np.searchsorted(cumulative, arc_m, side="right"))
    upper = min(max(upper, 1), len(trajectory) - 1)
    lower = upper - 1
    denom = max(float(cumulative[upper] - cumulative[lower]), 1e-8)
    ratio = (arc_m - float(cumulative[lower])) / denom

    start = np.array(
        [trajectory[lower].x, trajectory[lower].y],
        dtype=np.float32,
    )
    end = np.array(
        [trajectory[upper].x, trajectory[upper].y],
        dtype=np.float32,
    )
    return start + ratio * (end - start)


@dataclass(frozen=True)
class HumanCoreAnchors:
    indices: tuple[int, ...]
    arcs_m: np.ndarray
    reasons: tuple[tuple[str, ...], ...]
    cumulative_m: np.ndarray


def extract_human_core_anchors(
    trajectory: Sequence[Pose4D],
    landmark_centroids_xy: Sequence[Iterable[float]] = (),
    min_step_m: float = 1.0,
    rdp_tolerance_m: float = 2.5,
    yaw_keyframe_deg: float = 30.0,
    landmark_radius_m: float = 30.0,
) -> HumanCoreAnchors:
    """Extract sparse human decision anchors from route/view/landmark changes."""
    if len(trajectory) == 0:
        raise ValueError("trajectory must contain at least one pose")

    cumulative = _cumulative_xy_distance(trajectory)
    reasons: dict[int, set[str]] = {
        0: {"start"},
        len(trajectory) - 1: {"goal"},
    }
    if len(trajectory) <= 2:
        indices = tuple(sorted(reasons))
        return HumanCoreAnchors(
            indices,
            cumulative[list(indices)].copy(),
            tuple(tuple(sorted(reasons[idx])) for idx in indices),
            cumulative,
        )

    yaw_threshold = np.deg2rad(float(yaw_keyframe_deg))

    # Remove only points that are both spatially tiny and visually redundant.
    filtered_indices = [0]
    for idx in range(1, len(trajectory) - 1):
        previous = trajectory[filtered_indices[-1]]
        pose = trajectory[idx]
        step = np.hypot(pose.x - previous.x, pose.y - previous.y)
        yaw_change = _angle_distance(pose.yaw, previous.yaw)
        if step >= min_step_m or yaw_change >= yaw_threshold:
            filtered_indices.append(idx)
    filtered_indices.append(len(trajectory) - 1)

    filtered_xy = np.asarray(
        [
            [trajectory[idx].x, trajectory[idx].y]
            for idx in filtered_indices
        ],
        dtype=np.float32,
    )
    for local_idx in _rdp_indices(filtered_xy, float(rdp_tolerance_m)):
        original_idx = filtered_indices[local_idx]
        reasons.setdefault(original_idx, set()).add("turn")

    # Keep first-person observation changes even when XY barely changes.
    last_yaw_idx = filtered_indices[0]
    for original_idx in filtered_indices[1:-1]:
        if _angle_distance(
            trajectory[original_idx].yaw,
            trajectory[last_yaw_idx].yaw,
        ) >= yaw_threshold:
            reasons.setdefault(original_idx, set()).add("yaw")
            last_yaw_idx = original_idx

    # Preserve the human passage most closely associated with every referenced
    # landmark, but only when the demonstrated route actually comes near it.
    xy = np.asarray(
        [[pose.x, pose.y] for pose in trajectory],
        dtype=np.float32,
    )
    for landmark_xy in landmark_centroids_xy:
        landmark = np.asarray(landmark_xy, dtype=np.float32)
        if landmark.shape != (2,):
            continue
        distances = np.linalg.norm(xy - landmark[None, :], axis=1)
        nearest_idx = int(np.argmin(distances))
        if float(distances[nearest_idx]) <= float(landmark_radius_m):
            reasons.setdefault(nearest_idx, set()).add("landmark")

    indices = tuple(sorted(reasons))
    return HumanCoreAnchors(
        indices=indices,
        arcs_m=cumulative[list(indices)].copy(),
        reasons=tuple(tuple(sorted(reasons[idx])) for idx in indices),
        cumulative_m=cumulative,
    )


def build_next_core_anchor_target(
    trajectory: Sequence[Pose4D],
    core: HumanCoreAnchors,
    current_xy: Iterable[float],
    horizons_m: Sequence[float],
    min_lookahead_m: float = 8.0,
    current_arc_m: float | None = None,
) -> dict[str, np.ndarray | float | int | tuple[str, ...]]:
    """Return the next human intent anchor and path segment leading to it.

    The target trajectory ends at the next sparse human decision anchor rather
    than the final destination.  Fixed horizons beyond that anchor are masked,
    which prevents the path generator from receiving supervision for motion
    that belongs to a later decision segment.
    """
    if current_arc_m is None:
        current_arc, projection_error = _project_xy_to_path(
            current_xy,
            trajectory,
            core.cumulative_m,
        )
    else:
        current_arc = float(
            np.clip(current_arc_m, 0.0, core.cumulative_m[-1])
        )
        projection_error = 0.0

    min_anchor_arc = current_arc + float(min_lookahead_m)
    anchor_pos = len(core.indices) - 1
    for pos, arc_m in enumerate(core.arcs_m):
        if float(arc_m) >= min_anchor_arc - 1e-6:
            anchor_pos = pos
            break

    anchor_idx = int(core.indices[anchor_pos])
    anchor_arc = float(core.arcs_m[anchor_pos])
    # At the very end the chosen anchor may be behind the requested lookahead.
    anchor_arc = max(current_arc, anchor_arc)
    anchor_xy = np.asarray(
        [trajectory[anchor_idx].x, trajectory[anchor_idx].y],
        dtype=np.float32,
    )
    anchor_distance = max(0.0, anchor_arc - current_arc)

    targets = []
    valid = []
    for horizon in horizons_m:
        horizon = float(horizon)
        is_valid = horizon <= anchor_distance + 1e-6
        sample_arc = min(anchor_arc, current_arc + horizon)
        targets.append(
            _sample_xy_at_arc(
                trajectory,
                core.cumulative_m,
                sample_arc,
            )
        )
        valid.append(float(is_valid))

    targets.append(anchor_xy)
    valid.append(1.0)

    return {
        "anchor_xy": anchor_xy,
        "anchor_arc_m": np.float32(anchor_arc),
        "anchor_distance_m": np.float32(anchor_distance),
        "anchor_index": np.int64(anchor_idx),
        "anchor_reasons": core.reasons[anchor_pos],
        "trajectory_xy": np.asarray(targets, dtype=np.float32),
        "valid": np.asarray(valid, dtype=np.float32),
        "projection_error_m": np.float32(projection_error),
    }
