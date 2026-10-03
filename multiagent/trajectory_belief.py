"""Trajectory-belief utilities shared by training and closed-loop execution."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from multiagent.space import Point2D, Pose4D


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


def clamp_goal_to_selected_cell(
    predicted_goal: torch.Tensor,
    cell_ids: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Refine a selected coarse cell with a continuous goal prediction.

    The heatmap remains responsible for the discrete region.  The continuous
    goal head may only refine inside that region, preventing a disagreement
    between the two heads from sending the fallback trajectory to another cell.
    """
    rows = torch.div(
        cell_ids,
        grid_size,
        rounding_mode="floor",
    ).to(predicted_goal.dtype)
    cols = (cell_ids % grid_size).to(predicted_goal.dtype)
    lower = torch.stack((rows, cols), dim=-1) / float(grid_size)
    upper = torch.stack(
        (rows + 1.0, cols + 1.0),
        dim=-1,
    ) / float(grid_size)
    return torch.maximum(
        lower,
        torch.minimum(predicted_goal, upper),
    )


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


def compute_heatmap_statistics(
    probabilities: torch.Tensor,
    gt_ids: torch.Tensor,
    gt_xy: torch.Tensor,
    grid_size: int,
    map_meters: float,
) -> dict:
    """Return per-sample diagnostics for the shared 7x7 spatial belief."""
    probs = probabilities.detach()
    gt_ids = gt_ids.to(device=probs.device, dtype=torch.long)
    gt_xy = gt_xy.to(device=probs.device, dtype=probs.dtype)

    top1_conf, top1_ids = probs.max(dim=1)
    topk_ids = torch.topk(
        probs,
        k=min(3, probs.shape[1]),
        dim=1,
    ).indices
    gt_prob = probs.gather(1, gt_ids[:, None]).squeeze(1)
    gt_rank = (probs > gt_prob[:, None]).sum(dim=1).to(probs.dtype) + 1.0

    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / float(np.log(probs.shape[1]))

    pred_rows = torch.div(top1_ids, grid_size, rounding_mode="floor")
    pred_cols = top1_ids % grid_size
    gt_rows = torch.div(gt_ids, grid_size, rounding_mode="floor")
    gt_cols = gt_ids % grid_size
    cell_error = torch.sqrt(
        (pred_rows.to(probs.dtype) - gt_rows.to(probs.dtype)).square()
        + (pred_cols.to(probs.dtype) - gt_cols.to(probs.dtype)).square()
    )

    centers = grid_cell_centers(
        probs.shape[0],
        grid_size,
        probs.device,
        probs.dtype,
    )
    top1_centers = centers.gather(
        1,
        top1_ids[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    coarse_goal_error_m = (
        torch.linalg.norm(top1_centers - gt_xy, dim=-1)
        * float(map_meters)
    )

    return {
        "heatmap_top1_acc": (top1_ids == gt_ids).to(probs.dtype),
        "heatmap_top3_acc": (topk_ids == gt_ids[:, None]).any(dim=1).to(probs.dtype),
        "heatmap_gt_prob": gt_prob,
        "heatmap_top1_conf": top1_conf,
        "heatmap_entropy": entropy,
        "heatmap_gt_rank": gt_rank,
        "heatmap_cell_error": cell_error,
        "heatmap_coarse_goal_error_m": coarse_goal_error_m,
    }


def compute_trajectory_statistics(
    top1_ids: torch.Tensor,
    refined_endpoints: torch.Tensor,
    proposal_ids: torch.Tensor,
    proposal_endpoints: torch.Tensor,
    proposal_trajectories: torch.Tensor,
    gt_ids: torch.Tensor,
    gt_xy: torch.Tensor,
    gt_trajectory_targets: torch.Tensor,
    gt_trajectory_valid: torch.Tensor,
    map_meters: float,
) -> dict:
    """Return per-sample endpoint/first-waypoint diagnostics."""
    dtype = refined_endpoints.dtype
    gt_ids = gt_ids.to(device=refined_endpoints.device, dtype=torch.long)
    gt_xy = gt_xy.to(device=refined_endpoints.device, dtype=dtype)

    top1_endpoints = refined_endpoints.gather(
        1,
        top1_ids[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    top1_endpoint_error_m = (
        torch.linalg.norm(top1_endpoints - gt_xy, dim=-1)
        * float(map_meters)
    )

    proposal_endpoint_errors = (
        torch.linalg.norm(proposal_endpoints - gt_xy[:, None, :], dim=-1)
        * float(map_meters)
    )
    oracle_topk_endpoint_error_m = proposal_endpoint_errors.min(dim=1).values
    topk_gt_recall = (
        proposal_ids == gt_ids[:, None]
    ).any(dim=1).to(dtype)

    first_wp_error_m = (
        torch.linalg.norm(
            proposal_trajectories[:, 0, 0, :] - gt_trajectory_targets[:, 0, :],
            dim=-1,
        )
        * float(map_meters)
    )
    first_wp_valid = gt_trajectory_valid[:, 0].to(dtype)

    return {
        "trajectory_top1_endpoint_error_m": top1_endpoint_error_m,
        "trajectory_oracle_topk_endpoint_error_m": oracle_topk_endpoint_error_m,
        "trajectory_topk_gt_recall": topk_gt_recall,
        "trajectory_first_wp_error_m": first_wp_error_m,
        "trajectory_first_wp_valid": first_wp_valid,
    }


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
    k = min(int(top_k), grid_size * grid_size)

    # Prioritize true local maxima, but keep a finite fallback when there are
    # fewer than K peaks. Hard -inf masking previously produced arbitrary
    # invalid proposal ids in that case. Since probabilities are in [0, 1],
    # a +2 bonus guarantees that all local maxima rank ahead of fallback cells.
    flat_probs = field.flatten(1)
    ranking_scores = flat_probs + peak_mask.flatten(1).to(flat_probs.dtype) * 2.0
    _, ids = torch.topk(ranking_scores, k=k, dim=1)
    scores = torch.gather(flat_probs, dim=1, index=ids)

    rows = torch.div(ids, grid_size, rounding_mode="floor").to(probabilities.dtype)
    cols = (ids % grid_size).to(probabilities.dtype)
    endpoints = torch.stack(
        ((rows + 0.5) / grid_size, (cols + 0.5) / grid_size),
        dim=-1,
    )
    return ids, scores, endpoints


def prepare_teacher_rollout_path(
    trajectory: Sequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare the recorded human Pose4D path for metric teacher rollout."""
    poses = np.asarray(
        [[p.x, p.y, p.z, p.yaw] for p in trajectory],
        dtype=np.float32,
    )
    if len(poses) == 0:
        raise ValueError("teacher trajectory must contain at least one pose")

    if len(poses) == 1:
        cumulative = np.zeros(1, dtype=np.float32)
    else:
        segments = np.linalg.norm(poses[1:, :2] - poses[:-1, :2], axis=1)
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float32), np.cumsum(segments, dtype=np.float32))
        )
    return poses, cumulative


def _interpolate_angle(yaw0: float, yaw1: float, ratio: float) -> float:
    """Interpolate yaw along the shortest wrapped angular displacement."""
    delta = (float(yaw1) - float(yaw0) + np.pi) % (2.0 * np.pi) - np.pi
    return float((float(yaw0) + ratio * delta + np.pi) % (2.0 * np.pi) - np.pi)


def sample_teacher_pose_at_arc(
    poses: np.ndarray,
    cumulative: np.ndarray,
    arc_m: float,
) -> Pose4D:
    """Interpolate human x/y/z/yaw at a physical arc-length position."""
    if len(poses) == 1:
        p = poses[0]
        return Pose4D(float(p[0]), float(p[1]), float(p[2]), float(p[3]))

    arc_m = float(np.clip(arc_m, 0.0, cumulative[-1]))
    if arc_m >= float(cumulative[-1]) - 1e-8:
        p = poses[-1]
        return Pose4D(float(p[0]), float(p[1]), float(p[2]), float(p[3]))

    upper = int(np.searchsorted(cumulative, arc_m, side="right"))
    upper = min(max(upper, 1), len(poses) - 1)
    lower = upper - 1
    denom = max(float(cumulative[upper] - cumulative[lower]), 1e-8)
    ratio = (arc_m - float(cumulative[lower])) / denom

    xyz = poses[lower, :3] + ratio * (poses[upper, :3] - poses[lower, :3])
    yaw = _interpolate_angle(poses[lower, 3], poses[upper, 3], ratio)
    return Pose4D(float(xyz[0]), float(xyz[1]), float(xyz[2]), yaw)


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


def sample_fixed_horizon_targets_from_arc(
    points: np.ndarray,
    cumulative: np.ndarray,
    current_arc_m: float,
    horizons_m: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample stable physical horizons from a known path arc position."""
    current_arc = float(np.clip(current_arc_m, 0.0, cumulative[-1]))
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


def sample_fixed_horizon_targets(
    points: np.ndarray,
    cumulative: np.ndarray,
    current_xy: Iterable[float],
    horizons_m: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project an arbitrary state to the human path, then sample horizons."""
    current = np.asarray(current_xy, dtype=np.float32)
    current_arc = _project_to_polyline(current, points, cumulative)
    return sample_fixed_horizon_targets_from_arc(
        points,
        cumulative,
        current_arc,
        horizons_m,
    )


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
