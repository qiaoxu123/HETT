"""Compact 28x28 language-conditioned spatial belief field for HETT heatmap."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SpatialBeliefOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    spatial_features: torch.Tensor



def metric_gaussian_target(
    normalized_goals: torch.Tensor,
    *,
    field_size: int,
    sigma_m: float,
    map_meters: float,
) -> torch.Tensor:
    """Normalized Gaussian target using metric rather than cell-space distance."""
    if sigma_m <= 0 or map_meters <= 0:
        raise ValueError("sigma_m and map_meters must be positive")
    goals = torch.as_tensor(
        normalized_goals, dtype=torch.float32,
        device=normalized_goals.device if torch.is_tensor(normalized_goals) else None,
    ).clamp(0.0, 1.0)
    cell_m = float(map_meters) / float(field_size)
    coords_m = (
        torch.arange(field_size, device=goals.device, dtype=torch.float32) + 0.5
    ) * cell_m
    grid_rows_m, grid_cols_m = torch.meshgrid(
        coords_m, coords_m, indexing="ij"
    )
    # normalized goals are (x, y), while tensor maps are indexed [row=y, col=x].
    target_rows_m = goals[:, 1] * float(map_meters)
    target_cols_m = goals[:, 0] * float(map_meters)
    dist_sq_m = (
        (grid_rows_m.unsqueeze(0) - target_rows_m[:, None, None]) ** 2
        + (grid_cols_m.unsqueeze(0) - target_cols_m[:, None, None]) ** 2
    )
    target = torch.exp(-dist_sq_m / (2.0 * float(sigma_m) ** 2))
    return target / target.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)


def local_soft_argmax_xy(
    probabilities: torch.Tensor,
    peak_ids: torch.Tensor,
    *,
    window_size: int = 3,
) -> torch.Tensor:
    """Refine peak cells to continuous normalized world (x, y)."""
    if probabilities.ndim != 3 or probabilities.shape[1] != probabilities.shape[2]:
        raise ValueError("probabilities must have shape [B,H,H]")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if peak_ids.ndim != 1 or peak_ids.shape[0] != probabilities.shape[0]:
        raise ValueError("peak_ids must have shape [B]")

    batch, field_size, _ = probabilities.shape
    radius = window_size // 2
    peak_rows = torch.div(peak_ids, field_size, rounding_mode="floor")
    peak_cols = peak_ids % field_size

    rows = torch.arange(field_size, device=probabilities.device).view(1, -1, 1)
    cols = torch.arange(field_size, device=probabilities.device).view(1, 1, -1)
    window = (
        (rows - peak_rows.view(batch, 1, 1)).abs() <= radius
    ) & (
        (cols - peak_cols.view(batch, 1, 1)).abs() <= radius
    )

    weights = probabilities * window.to(probabilities.dtype)
    norm = weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    weights = weights / norm

    row_centers = (
        torch.arange(field_size, device=probabilities.device, dtype=probabilities.dtype)
        + 0.5
    ) / field_size
    col_centers = row_centers
    y = (weights * row_centers.view(1, -1, 1)).sum(dim=(1, 2))
    x = (weights * col_centers.view(1, 1, -1)).sum(dim=(1, 2))
    return torch.stack((x, y), dim=1)


def greedy_nms_topk(
    probabilities: torch.Tensor,
    *,
    top_k: int,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Return flattened greedy-NMS indices for each 2-D belief field."""
    if probabilities.ndim != 3 or probabilities.shape[1] != probabilities.shape[2]:
        raise ValueError("probabilities must have shape [B,H,H]")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    scores = probabilities.clone()
    batch, field_size, _ = scores.shape
    top_k = min(int(top_k), field_size * field_size)
    radius = kernel_size // 2
    rows = torch.arange(field_size, device=scores.device).view(1, -1, 1)
    cols = torch.arange(field_size, device=scores.device).view(1, 1, -1)
    selected = []
    for _ in range(top_k):
        ids = scores.flatten(1).argmax(dim=1)
        selected.append(ids)
        pick_rows = torch.div(ids, field_size, rounding_mode="floor")
        pick_cols = ids % field_size
        suppress = (
            (rows - pick_rows.view(batch, 1, 1)).abs() <= radius
        ) & (
            (cols - pick_cols.view(batch, 1, 1)).abs() <= radius
        )
        scores = scores.masked_fill(suppress, -torch.inf)
    return torch.stack(selected, dim=1)



def should_apply_reference_rerank(
    *,
    feedback: str,
    train_ml,
    disabled: bool,
) -> bool:
    """Return True only for student inference/evaluation rollouts.

    Training invokes both teacher and student rollouts with train_ml set.
    Reference reranking is a checkpoint-compatible inference post-process and
    must never alter the student trajectories used to optimize the model.
    """
    return feedback == "student" and train_ml is None and not disabled


def referenced_landmark_proximity_prior(
    nav_maps: torch.Tensor,
    *,
    field_size: int,
    dilation_steps: int = 3,
    decay: float = 0.7,
) -> torch.Tensor:
    """Build a conservative soft prior around instruction-referenced landmarks.

    The HETT map convention is [current, explored, global, referenced].  This
    helper uses only channel 3 and never reads the target position.  The mask is
    pooled to the dense belief resolution, then expanded by one cell per step
    with geometrically decaying support so that nearby targets are preferred
    without forcing the goal onto the landmark footprint itself.
    """
    if nav_maps.ndim != 4 or nav_maps.shape[1] < 4:
        raise ValueError("nav_maps must have shape [B,>=4,H,W]")
    if field_size < 1:
        raise ValueError("field_size must be positive")
    if dilation_steps < 0:
        raise ValueError("dilation_steps must be non-negative")
    if not (0.0 < decay <= 1.0):
        raise ValueError("decay must be in (0,1]")

    referenced = (nav_maps[:, 3:4] > 0).to(dtype=torch.float32)
    coarse = F.adaptive_max_pool2d(
        referenced, (field_size, field_size)
    ).squeeze(1)
    reached = coarse.bool()
    prior = coarse
    value = 1.0

    for _ in range(dilation_steps):
        expanded = F.max_pool2d(
            reached.float().unsqueeze(1),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(1).bool()
        ring = expanded & ~reached
        value *= float(decay)
        prior = torch.where(
            ring,
            torch.full_like(prior, value),
            prior,
        )
        reached = expanded

    return prior


def rerank_heatmap_topk_with_reference(
    probabilities: torch.Tensor,
    topk_ids: torch.Tensor,
    reference_prior: torch.Tensor,
    *,
    rerank_top_k: int = 4,
    prior_weight: float = 0.75,
    max_log_margin: float = 0.35,
    min_prior_gain: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservatively rerank only ambiguous Top-K heatmap hypotheses.

    The raw heatmap Top-1 is preserved unless all of the following hold:
      * another candidate within the first rerank_top_k has a better combined
        heatmap+reference score;
      * raw Top-1 is uncertain (small log-probability margin);
      * the replacement has materially stronger referenced-landmark support.

    Returns selected flattened ids and a boolean tensor indicating changed
    samples.  No ground-truth quantity is used by this function.
    """
    if probabilities.ndim != 3 or probabilities.shape[1] != probabilities.shape[2]:
        raise ValueError("probabilities must have shape [B,H,H]")
    if reference_prior.shape != probabilities.shape:
        raise ValueError("reference_prior shape must match probabilities")
    if topk_ids.ndim != 2 or topk_ids.shape[0] != probabilities.shape[0]:
        raise ValueError("topk_ids must have shape [B,K]")
    if rerank_top_k < 1:
        raise ValueError("rerank_top_k must be positive")
    if prior_weight < 0 or max_log_margin < 0 or min_prior_gain < 0:
        raise ValueError("rerank thresholds/weight must be non-negative")

    k = min(int(rerank_top_k), int(topk_ids.shape[1]))
    candidate_ids = topk_ids[:, :k]
    flat_prob = probabilities.flatten(1)
    flat_prior = reference_prior.flatten(1)
    candidate_prob = flat_prob.gather(1, candidate_ids).clamp_min(1e-12)
    candidate_prior = flat_prior.gather(1, candidate_ids)

    base_ids = candidate_ids[:, 0]
    if k == 1:
        return base_ids, torch.zeros_like(base_ids, dtype=torch.bool)

    log_prob = candidate_prob.log()
    combined = log_prob + float(prior_weight) * candidate_prior
    selected_local = combined.argmax(dim=1)
    proposed_ids = candidate_ids.gather(
        1, selected_local.unsqueeze(1)
    ).squeeze(1)

    runner_up = log_prob[:, 1:].amax(dim=1)
    raw_margin = log_prob[:, 0] - runner_up
    proposed_prior = candidate_prior.gather(
        1, selected_local.unsqueeze(1)
    ).squeeze(1)
    prior_gain = proposed_prior - candidate_prior[:, 0]
    has_reference_signal = candidate_prior.amax(dim=1) > 0
    changed = (
        (selected_local != 0)
        & has_reference_signal
        & (raw_margin <= float(max_log_margin))
        & (prior_gain >= float(min_prior_gain))
    )
    selected_ids = torch.where(changed, proposed_ids, base_ids)
    return selected_ids, changed


class CompactSpatialBelief(nn.Module):
    """Map + instruction -> dense 28x28 belief field.

    Input channels are ordered as:
      current view, explored area, global landmark contours,
      instruction-referenced landmark mask.
    """

    def __init__(
        self,
        *,
        input_channels: int = 4,
        field_size: int = 28,
        hidden_dim: int = 256,
        language_dim: int = 768,
        attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.field_size = int(field_size)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.language_projection = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.language_gate = nn.Parameter(torch.tensor(0.0))
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(
        self,
        maps: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor | None = None,
    ) -> SpatialBeliefOutput:
        if maps.ndim != 4 or maps.shape[1] != 4:
            raise ValueError("belief maps must have shape [B,4,H,W]")
        if language_tokens.ndim != 3:
            raise ValueError("language_tokens must have shape [B,L,D]")

        features = self.map_encoder(maps)
        features = F.adaptive_avg_pool2d(
            features, (self.field_size, self.field_size)
        )
        batch, channels, height, width = features.shape
        spatial = features.flatten(2).transpose(1, 2)
        language = self.language_projection(language_tokens)

        key_padding_mask = None
        if language_mask is not None:
            if language_mask.shape != language_tokens.shape[:2]:
                raise ValueError("language_mask shape mismatch")
            key_padding_mask = ~language_mask.bool()

        attended, _ = self.cross_attention(
            query=self.spatial_norm(spatial),
            key=language,
            value=language,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        spatial = spatial + torch.tanh(self.language_gate) * attended
        conditioned = spatial.transpose(1, 2).reshape(
            batch, channels, height, width
        )
        logits = self.decoder(conditioned).squeeze(1)
        probabilities = torch.softmax(logits.flatten(1), dim=-1).reshape_as(logits)
        return SpatialBeliefOutput(logits, probabilities, conditioned)
