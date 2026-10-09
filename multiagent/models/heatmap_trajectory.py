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

from .candidate_relation_selector import CandidateRelationSelector


@dataclass
class TrajectoryProposals:
    trajectories: torch.Tensor   # [B,K,M,T,2] normalized CityNav xy
    mode_logits: torch.Tensor    # [B,K,M], conditioned on each proposed goal
    joint_logits: torch.Tensor   # [B,K,M], heatmap log p(goal) + mode log p(path|goal)
    goal_xy: torch.Tensor        # [B,K,2], normalized CityNav xy
    goal_ids: torch.Tensor       # [B,K]
    stop_logits: torch.Tensor   # [B]
    candidate_logits: torch.Tensor  # [B,K], learned goal evidence before prior fusion


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


def select_goal_mode_indices(goal_mode_logits):
    """Select the most likely GOAL first, then the best mode for that goal.

    Flat argmax over [goal, mode] is incorrect for goal selection: it
    unfairly favors goals with peaked mode distributions, even when their
    total goal probability is lower. This accepts normalized joint log
    probabilities or unnormalized goal+conditional-mode scores.
    """
    if goal_mode_logits.ndim != 3:
        raise ValueError("goal/mode scores must have shape [B,K,M]")
    _, _, modes = goal_mode_logits.shape
    goals = torch.logsumexp(goal_mode_logits, dim=-1).argmax(-1)
    rows = torch.arange(goals.shape[0], device=goals.device)
    modes_selected = goal_mode_logits[rows, goals].argmax(-1)
    return goals * modes + modes_selected


def arrival_stop_targets(current_xy, target_xy, *, map_meters, success_radius_m):
    """Label arrival consistently with CityNav's official success radius."""
    if map_meters <= 0 or success_radius_m <= 0:
        raise ValueError("map_meters and success_radius_m must be positive")
    if current_xy.shape != target_xy.shape or current_xy.ndim != 2 or current_xy.shape[-1] != 2:
        raise ValueError("current and target positions must be [B,2]")
    return ((current_xy - target_xy).norm(dim=-1) * map_meters
            <= success_radius_m).to(current_xy.dtype)


def resample_teacher_suffix(trajectory, current_xy, *, map_name, bounds,
                            map_meters, steps, goal_xy=None):
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
    if goal_xy is not None:
        goal = np.asarray(goal_xy, dtype=np.float32).reshape(1, 2)
        if np.linalg.norm(suffix[-1] - goal[0]) > 1e-5:
            suffix = np.concatenate((suffix, goal), axis=0)
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



def fixed_local_anchor_paths(current_xy, goal_xy, *, modes=3, waypoints=8,
                             local_step_norm=20.0 / 410.0):
    """GT-free fixed local path anchors with a matched metric action budget.

    For goals farther than the common waypoint step, each path ends exactly
    one full step from the UAV (except map-boundary clipping). The three
    modes steer by -0.30/0/+0.30 rad; they differ in direction but NOT maximum
    distance. For goals within one step they all terminate at that goal to
    avoid near-goal detours. The fixed controller executes ONLY the endpoint.
    """
    if modes != 3 or waypoints < 2 or local_step_norm <= 0:
        raise ValueError("compact controller requires 3 modes, >=2 waypoints and a positive step")
    if current_xy.ndim != 2 or goal_xy.ndim != 3 or current_xy.shape[0] != goal_xy.shape[0]:
        raise ValueError("expected current [B,2] and goals [B,K,2]")
    delta = goal_xy - current_xy[:, None]
    distance = delta.norm(dim=-1, keepdim=True)
    direction = delta / distance.clamp_min(1e-6)
    travel = distance.clamp(max=local_step_norm)
    lateral = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
    angles = torch.tensor((-0.30, 0.0, 0.30),
                          dtype=goal_xy.dtype, device=goal_xy.device)
    # No detour near the goal: all anchors must reach the SAME goal whenever
    # it is within one controller step. This also prevents overshooting.
    angles = torch.where(
        distance[:, :, None, :] > local_step_norm,
        angles[None, None, :, None],
        torch.zeros_like(angles[None, None, :, None]),
    )
    forward = direction[:, :, None, :] * torch.cos(angles)
    sideways = lateral[:, :, None, :] * torch.sin(angles)
    endpoint_delta = (forward + sideways) * travel[:, :, None, :]
    t = torch.linspace(1 / waypoints, 1., waypoints,
                       device=goal_xy.device, dtype=goal_xy.dtype)
    # Use a curved intermediate shape, but preserve identical metric
    # endpoint displacement in all three modes. Inference uses endpoint only.
    path = (current_xy[:, None, None, None, :]
            + forward[:, :, :, None, :] * travel[:, :, None, None, :]
              * t[None, None, None, :, None]
            + sideways[:, :, :, None, :] * travel[:, :, None, None, :]
              * torch.sin(t * math.pi / 2)[None, None, None, :, None])
    path = torch.cat(
        (path[..., :-1, :],
         (current_xy[:, None, None, :] + endpoint_delta).unsqueeze(-2)),
        dim=-2)
    return path.clamp(0., 1.)


def goal_distance_soft_ranking_loss(proposals, target_xy, *, map_meters,
                                    temperature_m=20.0, active=None):
    """SBF-inspired distance-label CE over all proposed goals.

    The goal soft labels are supervision ONLY. Model scores/inputs do not
    receive GT. Even an all-far candidate pool contributes a relative label.
    """
    if map_meters <= 0 or temperature_m <= 0:
        raise ValueError("map_meters and temperature_m must be positive")
    if target_xy.shape != (proposals.goal_xy.shape[0], 2):
        raise ValueError("target_xy must be [B,2]")
    d_m = (proposals.goal_xy.detach() - target_xy[:, None]).norm(dim=-1) * map_meters
    target_probs = F.softmax(-d_m / temperature_m, dim=-1).detach()
    log_probs = torch.logsumexp(proposals.joint_logits, dim=-1)
    sample_loss = -(target_probs * log_probs).sum(-1)
    if active is None:
        return sample_loss.mean()
    mask = active.to(sample_loss.dtype)
    return (sample_loss * mask).sum() / mask.sum().clamp_min(1)


def local_path_soft_ranking_loss(proposals, target_xy, current_xy,
                                teacher_next_xy, *, map_meters,
                                temperature_m=5.0, positive_radius_m=20.0,
                                local_step_m=20.0, active=None):
    """Soft path classification for GT-near GOALS, not full GT-conditioned paths.

    Fixed anchors are scored by their local endpoint proximity to a causal
    human-route label and by remaining distance to the selected goal. Goal
    probabilities do not enter this loss, and no GT enters inference.
    """
    if min(map_meters, temperature_m, positive_radius_m, local_step_m) <= 0:
        raise ValueError("all distances/temperatures must be positive")
    if any(x.shape != current_xy.shape for x in (target_xy, teacher_next_xy)):
        raise ValueError("target, current and teacher local xy must all be [B,2]")
    d_m = (proposals.goal_xy.detach() - target_xy[:, None]).norm(dim=-1) * map_meters
    nearest_distance, nearest_id = d_m.min(-1)
    mask = nearest_distance <= positive_radius_m
    if active is not None:
        mask = mask & active.bool()
    batch = torch.arange(current_xy.shape[0], device=current_xy.device)
    waypoints = proposals.trajectories[batch, nearest_id, :, -1, :].detach()
    modes = proposals.mode_logits[batch, nearest_id]
    # Project a future human waypoint to the same local horizon. This label
    # is not fed into the scorer, and must never guide inference/control.
    human_delta = teacher_next_xy - current_xy
    human_norm = human_delta.norm(dim=-1, keepdim=True)
    human_local = current_xy + human_delta * (local_step_m / map_meters / human_norm.clamp_min(1e-6)).clamp(max=1.)
    human_error = (waypoints - human_local[:, None]).norm(dim=-1) * map_meters
    progress_error = (waypoints - proposals.goal_xy.detach()[batch, nearest_id, None]).norm(dim=-1) * map_meters
    costs = 0.7 * human_error + 0.3 * progress_error
    labels = F.softmax(-costs / temperature_m, dim=-1).detach()
    sample_loss = -(labels * F.log_softmax(modes, dim=-1)).sum(-1)
    weights = mask.to(sample_loss.dtype)
    return (sample_loss * weights).sum() / weights.sum().clamp_min(1), mask.sum()


class HeatmapTrajectoryHead(nn.Module):
    def __init__(self, *, feature_dim=256, hidden_dim=128, modes=3, waypoints=8,
                 curve_scale=0.25, language_dim=768):
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
        # Residual candidate scorer: initialized to zero, so existing HETT
        # heatmap priorities are unchanged until this head is trained.
        self.goal_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.goal_scorer[-1].weight)
        nn.init.zeros_(self.goal_scorer[-1].bias)
        # SBF-inspired mechanism: per-candidate evidence from contextual
        # instruction + individually matched landmark geometry + UAV history.
        # A zero-initialized gate preserves all old checkpoint behaviors.
        self.candidate_relation = CandidateRelationSelector(
            feature_dim=feature_dim, language_dim=language_dim, hidden_dim=96,
            attention_heads=4)
        # A small nonzero gate lets the relation encoder receive gradients on
        # the very first optimizer step. Exact zero delayed all encoder
        # learning until the scalar gate moved away from zero.
        self.relation_gate = nn.Parameter(torch.tensor(math.atanh(0.1)))
        self.stop = nn.Sequential(nn.Linear(feature_dim + 3, hidden_dim // 2),
                                  nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        # Starting with zeros preserves the simple straight/curved anchor paths.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def _generate(self, features, current_xy, heading_sc, goal_xy, goal_features=None):
        batch, k, _ = goal_xy.shape
        if goal_features is None:
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
        return paths, mode_logits, context

    def _generate_compact(self, features, current_xy, heading_sc, goal_xy, goal_features,
                          *, local_step_norm):
        batch, k, _ = goal_xy.shape
        pose = torch.cat((current_xy, heading_sc), dim=-1)
        context = self.context(torch.cat((
            goal_features, goal_xy - current_xy[:, None],
            pose[:, None, :].expand(-1, k, -1)
        ), dim=-1))
        modes = torch.arange(self.modes, device=goal_xy.device)
        # A linear score of context + mode_embedding cancels the context
        # under the mode softmax and cannot learn state-dependent path choice.
        # Elementwise interaction creates genuine goal/pose-dependent logits.
        mode_features = (context[:, :, None] *
                         torch.tanh(self.mode_embedding(modes))[None, None])
        logits = self.mode_score(mode_features).squeeze(-1)
        paths = fixed_local_anchor_paths(
            current_xy, goal_xy, modes=self.modes,
            waypoints=self.waypoints, local_step_norm=local_step_norm)
        return paths, logits, context

    def forward(self, features, goal_probabilities, current_xy, heading_sc,
                *, top_k=5, nms_kernel=3, teacher_goal=None,
                language_tokens=None, language_mask=None, landmark_xy=None,
                landmark_extent=None, landmark_valid=None,
                landmark_text_mask=None, history_xy=None,
                relation_enabled=True, selector_mode='joint', compact=False,
                local_step_m=20.0, map_meters=410.0):
        if current_xy.ndim != 2 or current_xy.shape[-1] != 2:
            raise ValueError("current_xy must be [B,2]")
        if heading_sc.shape != current_xy.shape:
            raise ValueError("heading sin/cos must be [B,2]")
        goals, ids = heatmap_endpoints(goal_probabilities, top_k, nms_kernel=nms_kernel)
        # Reuse one differentiable spatial sampling for trajectory generation
        # and candidate-language relation scoring (previously two grid_samples).
        goal_features = _gather_heatmap_features(features, goals)
        if compact:
            paths, logits, context = self._generate_compact(
                features, current_xy, heading_sc, goals, goal_features,
                local_step_norm=local_step_m / map_meters)
        else:
            paths, logits, context = self._generate(
                features, current_xy, heading_sc, goals, goal_features=goal_features)
        log_goal = torch.log(goal_probabilities.flatten(1).gather(1, ids).clamp_min(1e-8))
        relative_xy = goals - current_xy[:, None, :]
        candidate_logits = self.goal_scorer(
            torch.cat((context, relative_xy, log_goal.unsqueeze(-1)), dim=-1)
        ).squeeze(-1)
        # Do not infer landmark identity from an aggregate mask. Matching
        # named-anchor geometry is required for relation-conditioned scores.
        if relation_enabled and landmark_xy is not None:
            required = (language_tokens, landmark_extent, landmark_valid,
                        landmark_text_mask)
            if any(value is None for value in required):
                raise ValueError("relation selector needs language + complete named-landmark inputs")
            evidence = self.candidate_relation(
                goal_xy=goals,
                goal_features=goal_features,
                current_xy=current_xy,
                heading_sc=heading_sc,
                language_tokens=language_tokens,
                language_mask=language_mask,
                landmark_xy=landmark_xy,
                landmark_extent=landmark_extent,
                landmark_valid=landmark_valid,
                landmark_text_mask=landmark_text_mask,
                history_xy=history_xy,
            )
            candidate_logits = candidate_logits + torch.tanh(self.relation_gate) * evidence
        # Joint goal/path log-probabilities; no oracle target is used here.
        goal_log_probs = F.log_softmax(log_goal + candidate_logits, dim=-1)
        joint = goal_log_probs[:, :, None] + F.log_softmax(logits, dim=-1)
        here = _gather_heatmap_features(features, current_xy[:, None])[:, 0]
        current_confidence = goal_probabilities.flatten(1).amax(-1, keepdim=True)
        if selector_mode == 'joint':
            stop_goal_scores = goal_log_probs
        elif selector_mode == 'prior':
            stop_goal_scores = F.log_softmax(log_goal, dim=-1)
        else:
            raise ValueError("selector_mode must be prior or joint")
        # The arrival head must see the endpoint actually chosen by the
        # controller, not Top-1 if joint reranking selected another goal.
        chosen_goal = goals[torch.arange(goals.shape[0], device=goals.device),
                            stop_goal_scores.argmax(-1)]
        near_goal = (chosen_goal - current_xy).norm(dim=-1, keepdim=True)
        if compact:
            stop = logits.new_zeros(logits.shape[0])
        else:
            stop = self.stop(torch.cat((here, current_confidence, near_goal,
                                        heading_sc[:, :1]), dim=-1)).squeeze(-1)
        result = TrajectoryProposals(paths, logits, joint, goals, ids, stop, candidate_logits)
        if compact or teacher_goal is None:
            return result, None
        if teacher_goal.shape != current_xy.shape:
            raise ValueError("teacher_goal must have shape [B,2]")
        teacher_paths, teacher_logits, _ = self._generate(
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


def candidate_ranking_loss(proposals, target_xy, *, map_meters,
                           positive_radius_m, active=None):
    """Learn which predicted HETT candidate matches the goal (train only).

    Targets label the nearest candidate ONLY if it lies within the success
    radius; otherwise the sample has no positive candidate and is ignored.
    This avoids teaching the selector that an arbitrary far-away goal is valid.
    """
    if map_meters <= 0 or positive_radius_m <= 0:
        raise ValueError("map_meters and positive_radius_m must be positive")
    if target_xy.shape != (proposals.goal_xy.shape[0], 2):
        raise ValueError("target_xy must be [B,2]")
    d_m = (proposals.goal_xy - target_xy[:, None]).norm(dim=-1) * map_meters
    nearest_distance, nearest_id = d_m.min(dim=-1)
    valid = nearest_distance <= positive_radius_m
    if active is not None:
        valid = valid & active.to(device=valid.device, dtype=torch.bool)
    # logsumexp over modes recovers learned goal probability.
    log_probs = torch.logsumexp(proposals.joint_logits, dim=-1)
    per_sample = F.nll_loss(log_probs, nearest_id, reduction="none")
    return (per_sample * valid.to(per_sample.dtype)).sum() / valid.sum().clamp_min(1)


def candidate_trajectory_imitation_loss(proposals, target_xy, teacher_points,
                                        *, map_meters, positive_radius_m,
                                        active=None):
    """Train predicted-goal paths, never GT-conditioned paths, when covered."""
    if teacher_points.shape != (proposals.goal_xy.shape[0],
                                proposals.trajectories.shape[-2], 2):
        raise ValueError("teacher_points must be [B,T,2]")
    if target_xy.shape != (proposals.goal_xy.shape[0], 2):
        raise ValueError("target_xy must be [B,2]")
    if map_meters <= 0 or positive_radius_m <= 0:
        raise ValueError("map_meters and positive_radius_m must be positive")
    d_m = (proposals.goal_xy - target_xy[:, None]).norm(dim=-1) * map_meters
    nearest_distance, nearest_id = d_m.min(dim=-1)
    valid = nearest_distance <= positive_radius_m
    if active is not None:
        valid = valid & active.to(device=valid.device, dtype=torch.bool)
    rows = torch.arange(nearest_id.shape[0], device=nearest_id.device)
    paths = proposals.trajectories[rows, nearest_id]
    mode_logits = proposals.mode_logits[rows, nearest_id]
    return trajectory_imitation_loss((paths, mode_logits),
                                     teacher_points, active=valid)
