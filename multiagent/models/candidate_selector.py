"""Independent visual evidence selector for dense heatmap candidates.

The selector deliberately does not consume heatmap scores, referenced-landmark
masks, or ground-truth target information. Heatmap/SBF remains a proposal
mechanism; this module only asks whether currently observed RGB evidence can
discriminate among the proposed spatial hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def heatmap_ids_to_normalized_xy(
    candidate_ids: torch.Tensor,
    *,
    field_size: int,
) -> torch.Tensor:
    """Convert flattened [row, col] heatmap IDs to normalized world (x, y)."""
    if candidate_ids.ndim != 2:
        raise ValueError("candidate_ids must have shape [B,K]")
    if field_size < 1:
        raise ValueError("field_size must be positive")
    rows = torch.div(candidate_ids, field_size, rounding_mode="floor").to(torch.float32)
    cols = (candidate_ids % field_size).to(torch.float32)
    return torch.stack(
        (
            (cols + 0.5) / float(field_size),
            (rows + 0.5) / float(field_size),
        ),
        dim=-1,
    )


def sample_candidate_features_from_current_view(
    frame_features: torch.Tensor,
    candidate_xy: torch.Tensor,
    agent_xy: torch.Tensor,
    sin_yaw: torch.Tensor,
    cos_yaw: torch.Tensor,
    view_radius_m: torch.Tensor,
    *,
    map_meters: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample candidate-specific visual evidence from the current 7x7 feature map.

    CityNav normalized coordinates use x increasing east and normalized y
    increasing south, while world y increases north. The RGB crop is oriented
    with image top = agent forward and image left = agent left.

    Returns:
        candidate_features: [B,K,C]
        visible_mask: [B,K], true only for candidates inside the current view.
    """
    if frame_features.ndim == 3:
        batch, channels, flattened = frame_features.shape
        side = int(round(flattened ** 0.5))
        if side * side != flattened:
            raise ValueError("flattened frame feature map must be square")
        frame_features = frame_features.reshape(batch, channels, side, side)
    if frame_features.ndim != 4:
        raise ValueError("frame_features must have shape [B,C,H,W] or [B,C,H*W]")
    if candidate_xy.ndim != 3 or candidate_xy.shape[-1] != 2:
        raise ValueError("candidate_xy must have shape [B,K,2]")
    if agent_xy.shape != (candidate_xy.shape[0], 2):
        raise ValueError("agent_xy must have shape [B,2]")
    if map_meters <= 0:
        raise ValueError("map_meters must be positive")

    batch = candidate_xy.shape[0]
    for name, value in (
        ("sin_yaw", sin_yaw),
        ("cos_yaw", cos_yaw),
        ("view_radius_m", view_radius_m),
    ):
        if value.reshape(batch, -1).shape[1] != 1:
            raise ValueError(f"{name} must provide one scalar per batch element")

    sin_yaw = sin_yaw.reshape(batch, 1)
    cos_yaw = cos_yaw.reshape(batch, 1)
    radius = view_radius_m.reshape(batch, 1).to(candidate_xy).clamp_min(1e-3)

    delta = candidate_xy - agent_xy[:, None, :]
    world_dx = delta[..., 0] * float(map_meters)
    # normalized y increases downward/south, opposite to world +y.
    world_dy = -delta[..., 1] * float(map_meters)

    front = world_dx * cos_yaw + world_dy * sin_yaw
    left = -world_dx * sin_yaw + world_dy * cos_yaw

    # grid_sample coordinates: x=-1 left, +1 right; y=-1 top, +1 bottom.
    grid_x = -left / radius
    grid_y = -front / radius
    visible = (
        torch.isfinite(grid_x)
        & torch.isfinite(grid_y)
        & (grid_x.abs() <= 1.0)
        & (grid_y.abs() <= 1.0)
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        frame_features,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.squeeze(-1).transpose(1, 2)
    sampled = sampled * visible.unsqueeze(-1).to(sampled.dtype)
    return sampled, visible


class CandidateVisualSelector(nn.Module):
    """Candidate RGB-language matcher independent of the heatmap proposal score."""

    def __init__(
        self,
        *,
        visual_dim: int = 512,
        language_dim: int = 768,
        hidden_dim: int = 256,
        attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.language_projection = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_visual: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_visual.ndim != 3:
            raise ValueError("candidate_visual must have shape [B,K,D]")
        if language_tokens.ndim != 3:
            raise ValueError("language_tokens must have shape [B,L,D]")
        visual = self.visual_projection(candidate_visual)
        language = self.language_projection(language_tokens)

        key_padding_mask = None
        if language_mask is not None:
            if language_mask.shape != language_tokens.shape[:2]:
                raise ValueError("language_mask shape mismatch")
            key_padding_mask = ~language_mask.bool()

        attended, _ = self.cross_attention(
            query=self.query_norm(visual),
            key=language,
            value=language,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = self.output_norm(visual + attended)
        return self.score(fused).squeeze(-1)


def _masked_logits(
    logits: torch.Tensor,
    visible_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != visible_mask.shape:
        raise ValueError("logits and visible_mask shape mismatch")
    mask = visible_mask.bool()
    # Keep softmax numerically defined when no hypothesis is currently visible.
    safe_mask = mask.clone()
    no_visible = ~safe_mask.any(dim=1)
    if no_visible.any():
        safe_mask[no_visible, 0] = True
    return logits.masked_fill(~safe_mask, -torch.inf)


@dataclass(frozen=True)
class CandidateSelectionDecision:
    chosen_ids: torch.Tensor
    triggered: torch.Tensor
    confidence: torch.Tensor
    margin: torch.Tensor
    selected_rank: torch.Tensor
    visible_count: torch.Tensor


def select_candidate_with_abstention(
    raw_top1_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    selector_logits: torch.Tensor,
    visible_mask: torch.Tensor,
    *,
    min_visible_candidates: int = 2,
    min_confidence: float = 0.55,
    min_margin: float = 0.10,
) -> CandidateSelectionDecision:
    """Use visual ranking only when evidence is sufficiently discriminative."""
    if raw_top1_ids.ndim != 1:
        raise ValueError("raw_top1_ids must have shape [B]")
    if candidate_ids.shape != selector_logits.shape or candidate_ids.shape != visible_mask.shape:
        raise ValueError("candidate tensors must share shape [B,K]")
    if candidate_ids.shape[0] != raw_top1_ids.shape[0]:
        raise ValueError("batch mismatch")
    if min_visible_candidates < 1:
        raise ValueError("min_visible_candidates must be positive")

    masked = _masked_logits(selector_logits, visible_mask)
    probabilities = torch.softmax(masked, dim=1)
    top_values, top_indices = probabilities.topk(
        k=min(2, probabilities.shape[1]),
        dim=1,
    )
    confidence = top_values[:, 0]
    if top_values.shape[1] == 1:
        margin = confidence
    else:
        margin = top_values[:, 0] - top_values[:, 1]
    selected_rank = top_indices[:, 0]
    selected_ids = candidate_ids.gather(1, selected_rank[:, None]).squeeze(1)
    visible_count = visible_mask.sum(dim=1)

    triggered = (
        (visible_count >= int(min_visible_candidates))
        & (confidence >= float(min_confidence))
        & (margin >= float(min_margin))
    )
    chosen_ids = torch.where(triggered, selected_ids, raw_top1_ids)
    return CandidateSelectionDecision(
        chosen_ids=chosen_ids,
        triggered=triggered,
        confidence=confidence,
        margin=margin,
        selected_rank=selected_rank,
        visible_count=visible_count,
    )


@dataclass(frozen=True)
class CandidateRankingLoss:
    total: torch.Tensor
    hard: torch.Tensor
    listwise: torch.Tensor
    eligible_count: torch.Tensor
    nearest_index: torch.Tensor
    eligible_mask: torch.Tensor


def candidate_ranking_loss(
    selector_logits: torch.Tensor,
    candidate_distances_m: torch.Tensor,
    visible_mask: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    good_radius_m: float = 20.0,
    list_temperature_m: float = 20.0,
    list_weight: float = 0.5,
    min_visible_candidates: int = 2,
) -> CandidateRankingLoss:
    """Train Top-1 discrimination only when a good visible candidate exists."""
    if selector_logits.shape != candidate_distances_m.shape:
        raise ValueError("logits and distances shape mismatch")
    if selector_logits.shape != visible_mask.shape:
        raise ValueError("visible mask shape mismatch")
    if active_mask.ndim != 1 or active_mask.shape[0] != selector_logits.shape[0]:
        raise ValueError("active_mask must have shape [B]")
    if good_radius_m <= 0 or list_temperature_m <= 0:
        raise ValueError("distance hyperparameters must be positive")

    visible_bool = visible_mask.bool()
    visible_distances = candidate_distances_m.masked_fill(
        ~visible_bool, torch.inf
    )
    nearest_index = visible_distances.argmin(dim=1)
    nearest_distance = visible_distances.gather(
        1, nearest_index[:, None]
    ).squeeze(1)
    visible_count = visible_bool.sum(dim=1)
    eligible = (
        active_mask.bool()
        & (visible_count >= int(min_visible_candidates))
        & torch.isfinite(nearest_distance)
        & (nearest_distance <= float(good_radius_m))
    )

    masked_logits = _masked_logits(selector_logits, visible_mask)
    zero = selector_logits.sum() * 0.0
    if not eligible.any():
        return CandidateRankingLoss(
            total=zero,
            hard=zero,
            listwise=zero,
            eligible_count=eligible.sum(),
            nearest_index=nearest_index,
            eligible_mask=eligible,
        )

    eligible_logits = masked_logits[eligible]
    eligible_nearest = nearest_index[eligible]
    eligible_visible = visible_mask[eligible].bool()
    eligible_distances = candidate_distances_m[eligible]

    hard = F.cross_entropy(
        eligible_logits,
        eligible_nearest,
        reduction="sum",
    )

    distance_logits = (-eligible_distances / float(list_temperature_m)).masked_fill(
        ~eligible_visible,
        -torch.inf,
    )
    target = torch.softmax(distance_logits, dim=1)
    prediction_log = torch.log_softmax(eligible_logits, dim=1)
    listwise = -(target * prediction_log).masked_fill(
        ~eligible_visible, 0.0
    ).sum()
    total = hard + float(list_weight) * listwise
    return CandidateRankingLoss(
        total=total,
        hard=hard,
        listwise=listwise,
        eligible_count=eligible.sum(),
        nearest_index=nearest_index,
        eligible_mask=eligible,
    )
