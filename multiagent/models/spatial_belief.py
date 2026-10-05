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
