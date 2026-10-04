"""Static goal-prior model: language + static global map -> dense belief map."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StaticBeliefOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    spatial_features: torch.Tensor


def masked_spatial_softmax(logits: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Normalize logits over spatial cells while respecting invalid map padding."""
    if logits.ndim != 3:
        raise ValueError(f"expected [B,H,W] logits, got {tuple(logits.shape)}")
    flat = logits.flatten(1)
    if valid_mask is not None:
        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(0).expand_as(logits)
        if valid_mask.shape != logits.shape:
            raise ValueError("valid_mask shape must match logits")
        valid = valid_mask.bool().flatten(1)
        if (~valid.any(dim=1)).any():
            raise ValueError("every sample needs at least one valid spatial cell")
        flat = flat.masked_fill(~valid, -torch.inf)
    return torch.softmax(flat, dim=-1).reshape_as(logits)


class StaticBeliefModel(nn.Module):
    """Small spatial encoder with token-level language cross-attention.

    Expected inputs are static for an episode:
      * static_map: global landmark/semantic raster and optional start-state channels.
      * language_tokens: instruction token features from an existing text encoder.

    No RGB, trajectory history, explored-area history, or target-distance signal is used.
    """

    def __init__(
        self,
        *,
        input_channels: int = 4,
        language_dim: int = 768,
        hidden_dim: int = 256,
        attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")

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
        static_map: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor | None = None,
        valid_field_mask: torch.Tensor | None = None,
    ) -> StaticBeliefOutput:
        if static_map.ndim != 4:
            raise ValueError("static_map must have shape [B,C,H,W]")
        if language_tokens.ndim != 3:
            raise ValueError("language_tokens must have shape [B,L,D]")

        features = self.map_encoder(static_map)
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
        conditioned = spatial.transpose(1, 2).reshape(batch, channels, height, width)
        logits = self.decoder(conditioned).squeeze(1)

        if valid_field_mask is not None and valid_field_mask.shape[-2:] != logits.shape[-2:]:
            valid_field_mask = torch.nn.functional.interpolate(
                valid_field_mask.float().unsqueeze(1)
                if valid_field_mask.ndim == 3
                else valid_field_mask.float(),
                size=logits.shape[-2:],
                mode="nearest",
            ).squeeze(1).bool()

        probabilities = masked_spatial_softmax(logits, valid_field_mask)
        return StaticBeliefOutput(logits, probabilities, conditioned)
