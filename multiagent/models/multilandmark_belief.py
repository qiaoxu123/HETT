"""Multi-landmark spatial belief head for HETT.

The released HETT target head regresses one coordinate.  This module instead
keeps every instruction landmark separate and scores a dense set of map cells.
It is deliberately independent of the simulator so it can be unit-tested on
CPU and reused by offline target-localisation diagnostics.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def make_grid_centers(grid_size: int) -> torch.Tensor:
    """Return row-major cell centres in normalized map coordinates."""
    axis = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def target_cell_ids(normalized_goals: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Convert normalized ``[x, y]`` targets to row-major cell ids."""
    cells = torch.floor(normalized_goals * grid_size).long()
    cells = cells.clamp_(0, grid_size - 1)
    return cells[:, 0] * grid_size + cells[:, 1]


def target_region_mask(
    normalized_goals: torch.Tensor,
    grid_centers: torch.Tensor,
    radius_m: float,
    map_meters: float,
) -> torch.Tensor:
    """Mark cells whose centres fall inside the navigation success radius."""
    distances_m = torch.cdist(normalized_goals, grid_centers) * map_meters
    mask = distances_m <= radius_m

    # A very small radius or coarse grid must still provide one supervised cell.
    missing = ~mask.any(dim=1)
    if missing.any():
        nearest = distances_m[missing].argmin(dim=1)
        mask[missing, nearest] = True
    return mask


class MultiLandmarkBeliefHead(nn.Module):
    """Predict a multi-modal target belief and a local offset for every cell.

    Landmark pooling is permutation invariant.  Names and centres remain paired,
    while invalid padded slots are masked from both the context attention and
    the candidate-to-landmark relation scores.
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 96,
        grid_size: int = 41,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

        self.register_buffer("grid_centers", make_grid_centers(grid_size), persistent=True)

        self.landmark_coord = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
        )
        self.landmark_name = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.context_query = nn.Linear(d_model, d_model)
        self.landmark_key = nn.Linear(d_model, d_model)
        self.landmark_value = nn.Linear(d_model, d_model)
        self.context_norm = nn.LayerNorm(d_model)

        self.grid_hidden = nn.Linear(2, hidden_dim)
        self.context_hidden = nn.Linear(d_model, hidden_dim)
        self.landmark_hidden = nn.Linear(d_model, hidden_dim)
        self.geometry_hidden = nn.Linear(5, hidden_dim)

        self.base_score = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.relation_score = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )
        self.previous_belief_gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.previous_belief_gate.weight)
        nn.init.zeros_(self.previous_belief_gate.bias)

    def _landmark_context(
        self,
        context: torch.Tensor,
        landmark_tokens: torch.Tensor,
        landmark_valid: torch.Tensor,
        landmark_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.context_query(context).unsqueeze(1)
        keys = self.landmark_key(landmark_tokens)
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.d_model)
        if landmark_confidence is not None:
            confidence = landmark_confidence.to(scores.dtype).clamp(0.0, 1.0)
            scores = scores + torch.log(confidence.clamp_min(1e-6))
            landmark_valid = landmark_valid & (confidence > 0)
        scores = scores.masked_fill(~landmark_valid, -1e4)
        weights = torch.softmax(scores, dim=-1) * landmark_valid.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        landmark_context = (
            weights.unsqueeze(-1) * self.landmark_value(landmark_tokens)
        ).sum(dim=1)
        return self.context_norm(context + landmark_context)

    def forward(
        self,
        context: torch.Tensor,
        landmark_centers: torch.Tensor,
        landmark_name_features: torch.Tensor,
        landmark_valid: torch.Tensor,
        landmark_confidence: torch.Tensor | None = None,
        previous_belief: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``belief_logits, offsets, pred_goal, fused_context``.

        Shapes:
          context: ``[B, D]``
          landmark_centers: ``[B, N, 2]``
          landmark_name_features: ``[B, N, D]``
          landmark_valid: ``[B, N]``
          landmark_confidence: optional fuzzy-match confidence ``[B, N]``
          previous_belief: optional normalized log belief ``[B, G*G]``
        """
        landmark_valid = landmark_valid.bool()
        landmark_tokens = (
            self.landmark_coord(landmark_centers)
            + self.landmark_name(landmark_name_features)
        )
        fused_context = self._landmark_context(
            context, landmark_tokens, landmark_valid, landmark_confidence
        )

        batch_size, landmark_count, _ = landmark_centers.shape
        grid = self.grid_centers.to(dtype=context.dtype)
        grid_batch = grid.unsqueeze(0).expand(batch_size, -1, -1)

        grid_h = self.grid_hidden(grid_batch)
        context_h = self.context_hidden(fused_context).unsqueeze(1)
        base_hidden = grid_h + context_h
        evidence_logits = self.base_score(base_hidden).squeeze(-1)

        # Explicit candidate-to-landmark geometry.  The five values preserve
        # signed direction as well as distance without relying on landmark order.
        delta = grid_batch[:, :, None, :] - landmark_centers[:, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        unit = delta / distance.clamp_min(1e-6)
        geometry = torch.cat((delta, unit, distance), dim=-1)

        pair_hidden = (
            grid_h[:, :, None, :]
            + self.context_hidden(fused_context)[:, None, None, :]
            + self.landmark_hidden(landmark_tokens)[:, None, :, :]
            + self.geometry_hidden(geometry)
        )
        pair_logits = self.relation_score(pair_hidden).squeeze(-1)
        valid_float = landmark_valid[:, None, :].to(pair_logits.dtype)
        if landmark_confidence is not None:
            confidence = landmark_confidence[:, None, :].to(pair_logits.dtype)
            valid_float = valid_float * confidence.clamp(0.0, 1.0)
        relation_logits = (pair_logits * valid_float).sum(dim=-1)
        relation_logits = relation_logits / valid_float.sum(dim=-1).clamp_min(1.0)
        evidence_logits = evidence_logits + relation_logits
        current_log_belief = torch.log_softmax(evidence_logits, dim=-1)

        if previous_belief is not None:
            if previous_belief.shape != evidence_logits.shape:
                raise ValueError(
                    "previous_belief must have shape "
                    f"{tuple(evidence_logits.shape)}, got {tuple(previous_belief.shape)}"
                )
            gate = torch.sigmoid(self.previous_belief_gate(fused_context))
            previous_log_belief = torch.log_softmax(previous_belief, dim=-1)
            belief_logits = torch.logaddexp(
                torch.log(gate.clamp_min(1e-6)) + previous_log_belief,
                torch.log((1.0 - gate).clamp_min(1e-6)) + current_log_belief,
            )
        else:
            belief_logits = current_log_belief

        relation_hidden = (pair_hidden * valid_float[..., None]).sum(dim=2)
        relation_hidden = relation_hidden / valid_float.sum(dim=-1).clamp_min(1.0)[..., None]
        max_offset = 0.5 / self.grid_size
        offsets = self.offset_head(base_hidden + relation_hidden) * max_offset

        peak_ids = belief_logits.argmax(dim=-1)
        batch_ids = torch.arange(batch_size, device=context.device)
        pred_goal = grid_batch[batch_ids, peak_ids] + offsets[batch_ids, peak_ids]
        pred_goal = pred_goal.clamp(0.0, 1.0)
        return belief_logits, offsets, pred_goal, fused_context
