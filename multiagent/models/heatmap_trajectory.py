"""Lightweight goal-conditioned multimodal path proposals for HETT Heatmap.

No SBF components or oracle maps are used. During inference all endpoints
come from the learned HETT heatmap. During training, the *label-only* teacher
endpoint also conditions an auxiliary trajectory loss.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TrajectoryProposals:
    trajectories: torch.Tensor   # [B,K,M,T,2] normalized CityNav xy
    mode_logits: torch.Tensor    # [B,K,M], conditioned on each proposed goal
    joint_logits: torch.Tensor   # [B,K,M], heatmap log p(goal) + mode log p(path|goal)
    goal_xy: torch.Tensor        # [B,K,2], normalized CityNav xy
    goal_ids: torch.Tensor       # [B,K]
    stop_logits: torch.Tensor   # [B]


def heatmap_endpoints(probabilities, top_k, *, nms_kernel=3):
    from .spatial_belief import greedy_nms_topk
    if probabilities.ndim != 3 or probabilities.shape[-1] != probabilities.shape[-2]:
        raise ValueError("probabilities must be [B,H,H]")
    side = probabilities.shape[-1]
    ids = greedy_nms_topk(probabilities, top_k=top_k, kernel_size=nms_kernel)
    row = ids // side
    col = ids % side
    xy = torch.stack(((col + .5) / side, (row + .5) / side), dim=-1).to(probabilities.dtype)
    return xy, ids


def resample_teacher_suffix(trajectory, current_xy, *, map_name, bounds,
                            map_meters, steps):
    """Resample the *future teacher label* by arclength from nearest pose.

    This function must ONLY be invoked in the training target pipeline.
    It does not enter the model's inference inputs.
    """
    if steps <= 0 or map_meters <= 0:
        raise ValueError("steps/map_meters must be positive")
    if len(trajectory) == 0:
        raise ValueError("teacher trajectory is empty")
    mb = bounds[map_name]
    xy = np.asarray([[(p.xy.x - mb.x_min) / map_meters,
                      (mb.y_max - p.xy.y) / map_meters] for p in trajectory], dtype=np.float32)
    pos = np.asarray(current_xy, dtype=np.float32)
    nearest = int(np.linalg.norm(xy - pos, axis=-1).argmin())
    suffix = np.concatenate((pos[None], xy[nearest + 1:]), axis=0)
    if len(suffix) == 1:
        suffix = np.concatenate((suffix, suffix), axis=0)
    lengths = np.linalg.norm(np.diff(suffix, axis=0), axis=1)
    cumulative = np.concatenate(([0.], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-6:
        points = np.repeat(suffix[-1:], steps, axis=0)
    else:
        positions = np.linspace(0., cumulative[-1], steps + 1)[1:]
        keep = np.concatenate(([True], np.diff(cumulative) > 1e-6))
        points = np.stack((
            np.interp(positions, cumulative[keep], suffix[keep, 0]),
            np.interp(positions, cumulative[keep], suffix[keep, 1]),
        ), axis=-1)
    return torch.from_numpy(np.asarray(points, dtype=np.float32))


def _gather_heatmap_features(features, xy):
    """Differentiable sampling of HETT spatial features at candidate goals."""
    if features.ndim != 4 or xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("expected features [B,D,H,W], candidates [B,K,2]")
    grid = xy * 2 - 1
    samples = F.grid_sample(features, grid[:, :, None], align_corners=False,
                            mode="bilinear", padding_mode="border")
    return samples.squeeze(-1).transpose(1, 2)


class HeatmapTrajectoryHead(nn.Module):
    def __init__(self, *, feature_dim=256, hidden_dim=128, modes=3, waypoints=8,
                 curve_scale=0.25):
        super().__init__()
        if modes < 1 or waypoints < 2:
            raise ValueError("modes >= 1 and waypoints >= 2 are required")
        self.modes = modes
        self.waypoints = waypoints
        self.curve_scale = curve_scale
        self.context = nn.Sequential(nn.Linear(feature_dim + 6, hidden_dim),
                                     nn.LayerNorm(hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.mode_embedding = nn.Embedding(modes, hidden_dim)
        self.residual = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, waypoints * 2))
        self.mode_score = nn.Linear(hidden_dim, 1)
        self.stop = nn.Sequential(nn.Linear(feature_dim + 3, hidden_dim // 2),
                                  nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        # Starting with zeros preserves the simple straight/curved anchor paths.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def _generate(self, features, current_xy, heading_sc, goal_xy):
        batch, k, _ = goal_xy.shape
        goal_features = _gather_heatmap_features(features, goal_xy)
        pose = torch.cat((current_xy, heading_sc), dim=-1)
        context = self.context(torch.cat((
            goal_features, goal_xy - current_xy[:, None, :],
            pose[:, None, :].expand(-1, k, -1)
        ), dim=-1))
        mode_ids = torch.arange(self.modes, device=goal_xy.device)
        mode_features = context[:, :, None, :] + self.mode_embedding(mode_ids)[None, None]
        mode_logits = self.mode_score(mode_features).squeeze(-1)
        delta = goal_xy - current_xy[:, None]
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal = torch.stack((-delta[..., 1], delta[..., 0]), dim=-1) / distance
        t = torch.linspace(1 / self.waypoints, 1., self.waypoints,
                           device=goal_xy.device, dtype=goal_xy.dtype)
        baseline = current_xy[:, None, None, None, :] + (
            t[None, None, None, :, None] * delta[:, :, None, None, :]
        )
        offsets = torch.linspace(-1., 1., self.modes, device=goal_xy.device,
                                 dtype=goal_xy.dtype)
        envelope = torch.sin(math.pi * t)[None, None, None, :, None]
        curve = (offsets[None, None, :, None, None] * self.curve_scale
                 * distance[:, :, None, None, :] * envelope
                 * normal[:, :, None, None, :])
        residual = self.residual(mode_features).view(
            batch, k, self.modes, self.waypoints, 2)
        residual = torch.tanh(residual) * distance[:, :, None, None, :] * 0.2 * envelope
        paths = (baseline + curve + residual).clamp(0., 1.)
        # Final waypoint is always the proposed goal: makes endpoint semantics explicit.
        paths = torch.cat((paths[..., :-1, :], goal_xy[:, :, None, None, :].expand(
            -1, -1, self.modes, 1, -1)), dim=-2)
        return paths, mode_logits

    def forward(self, features, goal_probabilities, current_xy, heading_sc,
                *, top_k=5, nms_kernel=3, teacher_goal=None):
        if current_xy.ndim != 2 or current_xy.shape[-1] != 2:
            raise ValueError("current_xy must be [B,2]")
        if heading_sc.shape != current_xy.shape:
            raise ValueError("heading sin/cos must be [B,2]")
        goals, ids = heatmap_endpoints(goal_probabilities, top_k, nms_kernel=nms_kernel)
        paths, logits = self._generate(features, current_xy, heading_sc, goals)
        log_goal = torch.log(goal_probabilities.flatten(1).gather(1, ids).clamp_min(1e-8))
        joint = log_goal[:, :, None] + F.log_softmax(logits, dim=-1)
        here = _gather_heatmap_features(features, current_xy[:, None])[:, 0]
        current_confidence = goal_probabilities.flatten(1).amax(-1, keepdim=True)
        near_goal = (goals[:, 0] - current_xy).norm(dim=-1, keepdim=True)
        stop = self.stop(torch.cat((here, current_confidence, near_goal,
                                    heading_sc[:, :1]), dim=-1)).squeeze(-1)
        result = TrajectoryProposals(paths, logits, joint, goals, ids, stop)
        if teacher_goal is None:
            return result, None
        if teacher_goal.shape != current_xy.shape:
            raise ValueError("teacher_goal must have shape [B,2]")
        teacher_paths, teacher_logits = self._generate(
            features, current_xy, heading_sc, teacher_goal[:, None])
        return result, (teacher_paths[:, 0], teacher_logits[:, 0])


def trajectory_imitation_loss(teacher_prediction, teacher_points, *, active=None):
    """WTA imitation of a complete future path with mode classification."""
    paths, logits = teacher_prediction
    if paths.ndim != 4 or teacher_points.shape != (paths.shape[0], paths.shape[2], 2):
        raise ValueError("expected paths [B,M,T,2] and label [B,T,2]")
    distances = (paths - teacher_points[:, None]).abs().mean(dim=(-1, -2))
    best = distances.argmin(-1)
    chosen = distances.gather(1, best[:, None]).squeeze(1)
    selection = F.cross_entropy(logits, best, reduction="none")
    if active is None:
        active = torch.ones_like(chosen)
    active = active.to(dtype=chosen.dtype)
    denominator = active.sum().clamp_min(1)
    return ((chosen + .1 * selection) * active).sum() / denominator


def stop_supervision_loss(predicted, target, *, active=None, pos_weight=1.):
    loss = F.binary_cross_entropy_with_logits(
        predicted, target.to(predicted.dtype), reduction="none",
        pos_weight=predicted.new_tensor(pos_weight))
    if active is None:
        return loss.mean()
    valid = active.to(loss.dtype)
    return (loss * valid).sum() / valid.sum().clamp_min(1)
