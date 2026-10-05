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
    target_rows_m = goals[:, 0] * float(map_meters)
    target_cols_m = goals[:, 1] * float(map_meters)
    dist_sq_m = (
        (grid_rows_m.unsqueeze(0) - target_rows_m[:, None, None]) ** 2
        + (grid_cols_m.unsqueeze(0) - target_cols_m[:, None, None]) ** 2
    )
    target = torch.exp(-dist_sq_m / (2.0 * float(sigma_m) ** 2))
    return target / target.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)


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
