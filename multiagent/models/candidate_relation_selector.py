"""SBF-inspired *mechanism*, not an SBF model/weight transplant.

Condition every HETT goal proposal on contextual instruction tokens, aligned
referenced-landmark geometry, current pose, and previously observed UAV poses.
No target coordinates, teacher suffix, future frames or map identity enter.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def candidate_anchor_geometry(goal_xy, anchor_xy, anchor_extent):
    """Candidate-relative landmark geometry [B,K,M,8]; +y is world north."""
    if goal_xy.ndim != 3 or goal_xy.shape[-1] != 2:
        raise ValueError("goals must be [B,K,2]")
    if anchor_xy.ndim != 3 or anchor_xy.shape[-1] != 2:
        raise ValueError("anchors must be [B,M,2]")
    if anchor_extent.shape != anchor_xy.shape or anchor_xy.shape[0] != goal_xy.shape[0]:
        raise ValueError("anchor extent/batch mismatch")
    delta = goal_xy[:, :, None] - anchor_xy[:, None]
    east = delta[..., 0]
    north = -delta[..., 1]
    distance = torch.sqrt(east.square() + north.square())
    scale = distance.clamp_min(1e-6)
    width = anchor_extent[:, None, :, 0].expand_as(east)
    height = anchor_extent[:, None, :, 1].expand_as(east)
    in_extent = ((east.abs() <= width / 2) &
                 (north.abs() <= height / 2)).to(goal_xy.dtype)
    return torch.stack((east, north, distance, east / scale, north / scale,
                        width, height, in_extent), dim=-1)


def candidate_pose_history_features(goal_xy, current_xy, heading_sc, history_xy=None):
    """[B,K,7] geometry + causally observed history (only past/current poses)."""
    if current_xy.shape != (goal_xy.shape[0], 2) or heading_sc.shape != current_xy.shape:
        raise ValueError("current position/heading must be [B,2]")
    d = goal_xy - current_xy[:, None]
    east = d[..., 0]
    north = -d[..., 1]
    dist = torch.linalg.vector_norm(d, dim=-1).clamp_min(1e-6)
    yaw_sin = heading_sc[:, 0, None]
    yaw_cos = heading_sc[:, 1, None]
    relative_cos = (east * yaw_cos + north * yaw_sin) / dist
    relative_sin = (north * yaw_cos - east * yaw_sin) / dist
    history_near = torch.zeros_like(dist)
    made_progress = torch.zeros_like(dist)
    if history_xy is not None:
        if history_xy.ndim != 3 or history_xy.shape[0] != goal_xy.shape[0] or history_xy.shape[-1] != 2:
            raise ValueError("history_xy must be [B,T,2]")
        # Last state is the current UAV pose. Exclude it from visited evidence.
        if history_xy.shape[1] > 1:
            earlier = history_xy[:, :-1]
            previous_distance = (goal_xy[:, :, None] - earlier[:, None]).norm(dim=-1)
            history_near = previous_distance.min(dim=-1).values
            first_distance = (goal_xy - earlier[:, :1]).norm(dim=-1)
            made_progress = first_distance - dist
    return torch.stack((east, north, dist, relative_sin, relative_cos,
                        history_near, made_progress), dim=-1)


class CandidateRelationSelector(nn.Module):
    """Scores each goal using *its own* named-anchor and language relations.

    The per-candidate query attends to full contextual language. Landmark
    mention spans then bind text evidence to the landmark's exact geometry.
    Masked mean/soft-min preserve multi-anchor conjunction evidence. A zero
    output for landmark-free samples prevents spurious landmark selection.
    """

    def __init__(self, *, feature_dim=256, language_dim=768,
                 hidden_dim=96, attention_heads=4, dropout=0.1):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.query_proj = nn.Sequential(
            nn.Linear(feature_dim + 7, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU())
        self.token_proj = nn.Linear(language_dim, hidden_dim)
        self.name_proj = nn.Linear(language_dim, hidden_dim)
        self.geometry_proj = nn.Sequential(nn.Linear(8, hidden_dim),
                                           nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.text_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.pair_score = nn.Linear(hidden_dim, 1)
        self.score = nn.Sequential(
            nn.Linear(3 * hidden_dim + 2, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, *, goal_xy, goal_features, current_xy, heading_sc,
                language_tokens, language_mask, landmark_xy, landmark_extent,
                landmark_valid, landmark_text_mask, history_xy=None):
        batch, k, _ = goal_xy.shape
        if goal_features.shape[:2] != (batch, k):
            raise ValueError("goal feature mismatch")
        if language_tokens.ndim != 3 or language_tokens.shape[0] != batch:
            raise ValueError("language tensor mismatch")
        if landmark_xy.ndim != 3 or landmark_xy.shape[0] != batch or landmark_xy.shape[-1] != 2:
            raise ValueError("landmark shape mismatch")
        m = landmark_xy.shape[1]
        if landmark_extent.shape != landmark_xy.shape or landmark_valid.shape != (batch, m):
            raise ValueError("landmark extent/valid mismatch")
        if landmark_text_mask.shape != (batch, m, language_tokens.shape[1]):
            raise ValueError("landmark mention-mask mismatch")
        if m == 0:
            return goal_xy.new_zeros((batch, k))
        valid = landmark_valid.bool()
        if not valid.any():
            return goal_xy.new_zeros((batch, k))

        if language_mask is None:
            language_mask = torch.ones(language_tokens.shape[:2],
                                       dtype=torch.bool, device=language_tokens.device)
        if language_mask.shape != language_tokens.shape[:2]:
            raise ValueError("language mask mismatch")
        lang_valid = language_mask.bool().clone()
        # Avoid all-masked rows in cross attention.
        lang_valid[~lang_valid.any(-1), 0] = True

        pose_geometry = candidate_pose_history_features(
            goal_xy, current_xy, heading_sc, history_xy)
        query = self.query_proj(torch.cat((goal_features, pose_geometry), dim=-1))
        # Same projected tokens serve as keys and values. Avoid running the
        # identical learned projection twice per candidate-selection step.
        projected_tokens = self.token_proj(language_tokens)
        attended, _ = self.text_attention(
            query, projected_tokens, projected_tokens,
            key_padding_mask=~lang_valid, need_weights=False)

        mention_mask = landmark_text_mask.bool() & language_mask[:, None].bool()
        mention_weight = mention_mask.to(language_tokens.dtype)
        name_vectors = torch.einsum("bml,bld->bmd", mention_weight, language_tokens)
        name_vectors = name_vectors / mention_weight.sum(-1, keepdim=True).clamp_min(1)
        lang_weight = lang_valid.to(language_tokens.dtype)
        global_text = ((language_tokens * lang_weight[..., None]).sum(1)
                       / lang_weight.sum(1, keepdim=True).clamp_min(1))
        name_vectors = torch.where(mention_mask.any(-1)[..., None],
                                   name_vectors, global_text[:, None])
        anchor_geom = candidate_anchor_geometry(
            goal_xy, landmark_xy, landmark_extent)
        pair = self.pair_norm(
            query[:, :, None] + attended[:, :, None]
            + self.name_proj(name_vectors)[:, None]
            + self.geometry_proj(anchor_geom))

        per_anchor = self.pair_score(pair).squeeze(-1)
        active = valid[:, None, :].expand(batch, k, m)
        counts = active.sum(-1).clamp_min(1)
        mean = (per_anchor * active.to(per_anchor.dtype)).sum(-1) / counts
        temperature = 0.5
        minimum = -temperature * (
            torch.logsumexp((-per_anchor / temperature).masked_fill(
                ~active, -torch.inf), dim=-1) - counts.to(per_anchor.dtype).log())
        # Some episodes have no matched landmarks even when others in the
        # batch do. Avoid inf * 0 -> NaN for their masked soft-min.
        minimum = torch.where(active.any(-1), minimum, torch.zeros_like(minimum))
        # Learned attention over valid anchored evidence (not raw camera pixels).
        attn = torch.einsum("bkd,bkmd->bkm", query, pair) / math.sqrt(pair.shape[-1])
        attn = F.softmax(attn.masked_fill(~active, -1e4), dim=-1)
        attn = attn * active.to(attn.dtype)
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (pair * attn[..., None]).sum(dim=-2)
        combined = torch.cat((query, attended, pooled,
                              mean[..., None], minimum[..., None]), dim=-1)
        scores = self.score(combined).squeeze(-1)
        return scores * valid.any(-1).to(scores.dtype)[:, None]
