"""Utilities for decoupled target-localisation and action training.

The target stage learns a stable global coordinate from deployment-available
inputs.  The action stage treats that coordinate as a detached condition and
only updates a small action-specific adapter plus the direction/progress heads.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    elif stage == "action":
        for model in (language_model, vision_model, navigation_model):
            for parameter in model.parameters():
                parameter.requires_grad = False
        for module in action_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
