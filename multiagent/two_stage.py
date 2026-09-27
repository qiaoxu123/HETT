"""Utilities for decoupled target-localisation and action training.

The target stage learns a stable global coordinate from deployment-available
inputs.  The action stage treats that coordinate as a detached condition and
only updates a small action-specific adapter plus the direction/progress heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


TRAINING_STAGES = ("joint", "target", "action")
ACTION_MODULE_NAMES = (
    "target_conditioning",
    "decoder_2_action_full",
    "decoder_2_progress_full",
)


@dataclass
class TargetLosses:
    coordinate: torch.Tensor
    distance: torch.Tensor
    bearing: torch.Tensor
    error_m: torch.Tensor


@dataclass
class QuadtreeBelief:
    """Sparse hierarchical target belief decoded from a quadtree beam."""

    coordinate: torch.Tensor
    leaf_centers: torch.Tensor
    leaf_probs: torch.Tensor
    hierarchy_loss: Optional[torch.Tensor]


class SparseQuadtreeBeliefHead(nn.Module):
    """Hierarchical sparse target localizer for normalized 2-D maps.

    Training uses teacher-forced 4-way decisions along the ground-truth path,
    so the classification cost is O(4 * depth). Inference keeps only the
    highest-scoring top-k nodes at every level, giving O(4 * top-k * depth)
    spatial hypotheses instead of scoring a dense HxW field.

    The final coordinate is the probability-weighted mean of the retained
    leaves. The leaf centers/probabilities are also returned so a later
    controller can consume multiple target hypotheses without early collapse.
    """

    _OFFSETS = (
        (-1.0, -1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (1.0, 1.0),
    )

    def __init__(
        self,
        d_model: int,
        depth: int = 5,
        topk: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("quadtree depth must be positive")
        if topk < 1:
            raise ValueError("quadtree topk must be positive")
        self.depth = depth
        self.topk = topk
        self.context_proj = nn.Linear(d_model, hidden_dim)
        self.box_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _children(
        centers: torch.Tensor,
        sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand square nodes into four children."""

        offsets = centers.new_tensor(SparseQuadtreeBeliefHead._OFFSETS)
        child_sizes = sizes.unsqueeze(-1).expand(*sizes.shape, 4) * 0.5
        child_centers = (
            centers.unsqueeze(-2)
            + offsets * (sizes.unsqueeze(-1).unsqueeze(-1) * 0.25)
        )
        return child_centers.clamp(0.0, 1.0), child_sizes

    def _score_children(
        self,
        context: torch.Tensor,
        child_centers: torch.Tensor,
        child_sizes: torch.Tensor,
    ) -> torch.Tensor:
        context_h = self.context_proj(context)
        while context_h.ndim < child_centers.ndim:
            context_h = context_h.unsqueeze(-2)
        geometry = torch.cat(
            (child_centers, child_sizes.unsqueeze(-1)), dim=-1
        )
        hidden = context_h + self.box_proj(geometry)
        return self.fusion(hidden).squeeze(-1)

    @staticmethod
    def _truth_child(
        truth: torch.Tensor,
        parent_centers: torch.Tensor,
    ) -> torch.Tensor:
        x_right = (truth[:, 0] >= parent_centers[:, 0]).long()
        y_bottom = (truth[:, 1] >= parent_centers[:, 1]).long()
        return x_right * 2 + y_bottom

    def hierarchy_loss(
        self,
        context: torch.Tensor,
        truth: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced hierarchical CE along the ground-truth path."""

        batch = context.shape[0]
        parent_centers = context.new_full((batch, 2), 0.5)
        parent_sizes = context.new_ones(batch)
        loss = context.new_zeros(())
        for _ in range(self.depth):
            child_centers, child_sizes = self._children(
                parent_centers, parent_sizes
            )
            logits = self._score_children(context, child_centers, child_sizes)
            labels = self._truth_child(truth, parent_centers)
            loss = loss + F.cross_entropy(logits, labels, reduction="sum")
            gather_index = labels.view(batch, 1, 1).expand(-1, 1, 2)
            parent_centers = child_centers.gather(
                1, gather_index
            ).squeeze(1)
            parent_sizes = child_sizes.gather(
                1, labels.view(batch, 1)
            ).squeeze(1)
        return loss

    def decode(
        self,
        context: torch.Tensor,
        topk: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Beam-decode a sparse set of quadtree leaves."""

        batch = context.shape[0]
        keep = self.topk if topk is None else topk
        centers = context.new_full((batch, 1, 2), 0.5)
        sizes = context.new_ones(batch, 1)
        log_scores = context.new_zeros(batch, 1)

        for _ in range(self.depth):
            child_centers, child_sizes = self._children(centers, sizes)
            bsz, beam, _, _ = child_centers.shape
            flat_centers = child_centers.reshape(bsz, beam * 4, 2)
            flat_sizes = child_sizes.reshape(bsz, beam * 4)

            logits = self._score_children(
                context.unsqueeze(1).expand(-1, beam, -1),
                child_centers,
                child_sizes,
            )
            child_log_probs = F.log_softmax(logits, dim=-1)
            flat_scores = (
                log_scores.unsqueeze(-1) + child_log_probs
            ).reshape(bsz, beam * 4)

            next_keep = min(keep, flat_scores.shape[1])
            log_scores, indices = torch.topk(
                flat_scores, k=next_keep, dim=-1
            )
            centers = flat_centers.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, 2)
            )
            sizes = flat_sizes.gather(1, indices)

        probs = torch.softmax(log_scores, dim=-1)
        coordinate = (centers * probs.unsqueeze(-1)).sum(dim=1)
        return coordinate, centers, probs

    def forward(
        self,
        context: torch.Tensor,
        truth: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
    ) -> QuadtreeBelief:
        coordinate, centers, probs = self.decode(context, topk=topk)
        loss = None if truth is None else self.hierarchy_loss(context, truth)
        return QuadtreeBelief(
            coordinate=coordinate,
            leaf_centers=centers,
            leaf_probs=probs,
            hierarchy_loss=loss,
        )


class TargetConditioning(nn.Module):
    """Inject a global target coordinate into motion tokens.

    In addition to absolute ``(x, y)``, the embedding receives relative
    displacement, distance and unit direction.  Every value is derived from
    the supplied coordinate and current pose, so no extra privileged input is
    introduced.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(7, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.attention = nn.MultiheadAttention(
            d_model, num_heads=1, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    @staticmethod
    def geometry(
        target_coordinates: torch.Tensor,
        current_positions: torch.Tensor,
    ) -> torch.Tensor:
        delta = target_coordinates - current_positions
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        unit_delta = delta / distance.clamp_min(1e-6)
        return torch.cat(
            (target_coordinates, delta, distance, unit_delta), dim=-1
        )

    def forward(
        self,
        motion_tokens: torch.Tensor,
        target_coordinates: torch.Tensor,
        current_positions: torch.Tensor,
    ) -> torch.Tensor:
        target_token = self.projection(
            self.geometry(target_coordinates, current_positions)
        ).unsqueeze(1)
        delta, _ = self.attention(
            query=motion_tokens,
            key=target_token,
            value=target_token,
            need_weights=False,
        )
        motion_tokens = self.norm(motion_tokens + delta)
        return self.ffn_norm(motion_tokens + self.ffn(motion_tokens))


def target_coordinate_losses(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    current_positions: torch.Tensor,
    map_meters: float,
    huber_delta_m: float,
) -> TargetLosses:
    """Return metre-scaled coordinate, range and bearing losses.

    Inputs use the normalized map coordinate system.  The main coordinate loss
    is computed in metres so it cannot be numerically drowned by normalized
    action losses as easily as the released coordinate MSE.
    """

    predicted_m = predicted * map_meters
    truth_m = truth * map_meters
    current_m = current_positions * map_meters

    coordinate = F.huber_loss(
        predicted_m,
        truth_m,
        delta=huber_delta_m,
        reduction="sum",
    )
    pred_delta = predicted_m - current_m
    truth_delta = truth_m - current_m
    pred_distance = torch.linalg.vector_norm(pred_delta, dim=-1)
    truth_distance = torch.linalg.vector_norm(truth_delta, dim=-1)
    distance = F.huber_loss(
        pred_distance,
        truth_distance,
        delta=huber_delta_m,
        reduction="sum",
    )
    pred_unit = pred_delta / pred_distance.unsqueeze(-1).clamp_min(1e-6)
    truth_unit = truth_delta / truth_distance.unsqueeze(-1).clamp_min(1e-6)
    bearing = (1.0 - (pred_unit * truth_unit).sum(dim=-1)).sum()
    error_m = torch.linalg.vector_norm(predicted_m - truth_m, dim=-1)
    return TargetLosses(coordinate, distance, bearing, error_m)


def coordinate_gt_probability(
    current_epoch: int,
    total_epochs: int,
    start_probability: float,
    end_probability: float,
) -> float:
    """Linear curriculum from oracle coordinates to predicted coordinates."""

    if total_epochs <= 1:
        fraction = 1.0
    else:
        fraction = min(max((current_epoch - 1) / (total_epochs - 1), 0.0), 1.0)
    return start_probability + fraction * (end_probability - start_probability)


def select_action_coordinates(
    predicted: torch.Tensor,
    truth: torch.Tensor | None,
    use_truth: torch.Tensor | None,
) -> torch.Tensor:
    """Select detached coordinates used by the action branch."""

    predicted = predicted.detach()
    if truth is None or use_truth is None:
        return predicted
    if use_truth.ndim == 1:
        use_truth = use_truth.unsqueeze(-1)
    return torch.where(use_truth.bool(), truth.detach(), predicted)


def configure_stage_parameters(
    language_model: nn.Module,
    vision_model: nn.Module,
    navigation_model: nn.Module,
    stage: str,
    freeze_target_backbones: bool = False,
) -> None:
    """Configure strict parameter ownership for a training stage."""

    if stage not in TRAINING_STAGES:
        raise ValueError(f"unsupported training stage: {stage}")

    for model in (language_model, vision_model, navigation_model):
        for parameter in model.parameters():
            parameter.requires_grad = True

    action_modules = [getattr(navigation_model, name) for name in ACTION_MODULE_NAMES]
    if stage == "joint":
        # The module is not on the released joint forward path. Excluding it
        # also preserves old optimizer parameter groups for exact resumes.
        for parameter in navigation_model.target_conditioning.parameters():
            parameter.requires_grad = False
    elif stage == "target":
        for module in action_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
        if freeze_target_backbones:
            for model in (language_model, vision_model):
                for parameter in model.parameters():
                    parameter.requires_grad = False
    elif stage == "action":
        for model in (language_model, vision_model, navigation_model):
            for parameter in model.parameters():
                parameter.requires_grad = False
        for module in action_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
