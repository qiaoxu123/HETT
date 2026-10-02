"""Trajectory-belief utilities shared by training and closed-loop execution."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from multiagent.space import Point2D


def grid_cell_centers(
    batch_size: int,
    grid_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized centers for the discrete HETT grid."""
    coords = (
        torch.arange(grid_size, device=device, dtype=dtype) + 0.5
    ) / grid_size
    rows, cols = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack((rows, cols), dim=-1).reshape(1, -1, 2)
    return centers.expand(batch_size, -1, -1)


def refine_candidate_endpoints(
    endpoint_offsets: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Convert per-cell offsets into continuous normalized endpoints."""
    centers = grid_cell_centers(
        endpoint_offsets.shape[0],
        grid_size,
        endpoint_offsets.device,
        endpoint_offsets.dtype,
    )
    return (centers + endpoint_offsets).clamp(0.0, 1.0)


def build_fixed_horizon_anchors(
    current_xy: torch.Tensor,
    endpoints: torch.Tensor,
    horizons_m: Sequence[float],
    map_meters: float,
) -> torch.Tensor:
    """Straight anchors with fixed physical-distance semantics plus endpoint."""
    delta = endpoints - current_xy[:, None, :]
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
    unit = delta / distance.clamp_min(1e-8)

    horizons = torch.as_tensor(
        horizons_m,
        device=endpoints.device,
        dtype=endpoints.dtype,
    ) / float(map_meters)
    step_distance = torch.minimum(
        distance.unsqueeze(-2),
        horizons.view(1, 1, -1, 1),
    )
    intermediate = (
        current_xy[:, None, None, :]
        + unit.unsqueeze(-2) * step_distance
    )
    return torch.cat((intermediate, endpoints[:, :, None, :]), dim=2)


def select_nms_topk(
    probabilities: torch.Tensor,
    grid_size: int,
    top_k: int,
    kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-only local-max NMS followed by Top-K selection."""
    if kernel_size % 2 != 1:
        raise ValueError("trajectory_nms_kernel must be odd")
    batch_size = probabilities.shape[0]
    field = probabilities.view(batch_size, 1, grid_size, grid_size)
    pooled = F.max_pool2d(
        field,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    peak_mask = field >= pooled
    candidate_scores = field.masked_fill(~peak_mask, -torch.inf).flatten(1)
    k = min(int(top_k), grid_size * grid_size)
    scores, ids = torch.topk(candidate_scores, k=k, dim=1)

    rows = torch.div(ids, grid_size, rounding_mode="floor").to(probabilities.dtype)
    cols = (ids % grid_size).to(probabilities.dtype)
    endpoints = torch.stack(
        ((rows + 0.5) / grid_size, (cols + 0.5) / grid_size),
        dim=-1,
    )
    return ids, scores, endpoints


def prepare_teacher_path(
    trajectory: Sequence,
    goal_xy: Iterable[float],
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray([[p.x, p.y] for p in trajectory], dtype=np.float32)
    goal = np.asarray(goal_xy, dtype=np.float32)
    if len(points) == 0:
        points = goal[None]
    elif np.linalg.norm(points[-1] - goal) > 1e-4:
        points = np.concatenate((points, goal[None]), axis=0)

    if len(points) == 1:
        cumulative = np.zeros(1, dtype=np.float32)
    else:
        segments = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float32), np.cumsum(segments, dtype=np.float32))
        )
    return points, cumulative


def _project_to_polyline(
    current_xy: np.ndarray,
    points: np.ndarray,
    cumulative: np.ndarray,
) -> float:
    if len(points) <= 1:
        return 0.0

    starts = points[:-1]
    deltas = points[1:] - starts
    lengths_sq = np.sum(deltas * deltas, axis=1)
    safe = np.maximum(lengths_sq, 1e-8)
    ratios = np.sum((current_xy[None] - starts) * deltas, axis=1) / safe
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = starts + ratios[:, None] * deltas
    errors = np.linalg.norm(projections - current_xy[None], axis=1)
    idx = int(np.argmin(errors))
    segment_length = float(np.sqrt(lengths_sq[idx]))
    return float(cumulative[idx] + ratios[idx] * segment_length)


def _sample_at_arc(
    points: np.ndarray,
    cumulative: np.ndarray,
    arc: float,
) -> np.ndarray:
    if len(points) == 1:
        return points[0]
    arc = float(np.clip(arc, 0.0, cumulative[-1]))
    if arc >= cumulative[-1] - 1e-8:
        return points[-1]
    upper = int(np.searchsorted(cumulative, arc, side="right"))
    upper = min(max(upper, 1), len(points) - 1)
    lower = upper - 1
    denom = max(float(cumulative[upper] - cumulative[lower]), 1e-8)
    ratio = (arc - float(cumulative[lower])) / denom
    return points[lower] + ratio * (points[upper] - points[lower])


def sample_fixed_horizon_targets(
    points: np.ndarray,
    cumulative: np.ndarray,
    current_xy: Iterable[float],
    horizons_m: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample human teacher path at stable metric horizons plus final goal."""
    current = np.asarray(current_xy, dtype=np.float32)
    current_arc = _project_to_polyline(current, points, cumulative)
    remaining = max(0.0, float(cumulative[-1]) - current_arc)

    targets = []
    valid = []
    for horizon in horizons_m:
        horizon = float(horizon)
        targets.append(
            _sample_at_arc(points, cumulative, current_arc + min(horizon, remaining))
        )
        valid.append(horizon <= remaining + 1e-6)

    targets.append(points[-1])
    valid.append(True)
    return np.asarray(targets, dtype=np.float32), np.asarray(valid, dtype=np.float32)


def normalize_targets(env, map_name: str, map_meters: float, targets: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            env.normalize_position(
                Point2D(float(x), float(y)),
                map_name,
                map_meters,
            )
            for x, y in targets
        ],
        dtype=np.float32,
    )
