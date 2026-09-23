"""Masked losses for active-altitude auxiliary heads."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActiveAltitudeLosses:
    total: torch.Tensor
    altitude_action: torch.Tensor
    phase: torch.Tensor
    stop: torch.Tensor
    uncertainty: torch.Tensor


def active_altitude_losses(outputs: dict[str, torch.Tensor],
                           altitude_action_target: torch.Tensor,
                           phase_target: torch.Tensor,
                           stop_target: torch.Tensor,
                           goal_error_squared: torch.Tensor,
                           active: torch.Tensor,
                           altitude_weight: float = 0.5,
                           phase_weight: float = 0.25,
                           stop_weight: float = 1.0,
                           uncertainty_weight: float = 0.05) -> ActiveAltitudeLosses:
    """Compute losses only for episodes that have not already terminated."""
    if not bool(active.any()):
        zero = outputs["stop_logit"].sum() * 0.0
        return ActiveAltitudeLosses(zero, zero, zero, zero, zero)
    altitude_logits = outputs["altitude_action_logits"][active]
    altitude_targets = altitude_action_target[active]
    if altitude_targets.ndim == altitude_logits.ndim:
        altitude = -(altitude_targets * F.log_softmax(altitude_logits, dim=-1)).sum(-1).mean()
    else:
        altitude = F.cross_entropy(altitude_logits, altitude_targets)
    phase = F.cross_entropy(outputs["phase_logits"][active], phase_target[active])
    stop = F.binary_cross_entropy_with_logits(
        outputs["stop_logit"].reshape(-1)[active], stop_target.reshape(-1)[active].float())

    log_variance = outputs["goal_log_variance"].reshape(-1)[active]
    squared_error = goal_error_squared.reshape(-1)[active].detach()
    uncertainty = (torch.exp(-log_variance) * squared_error + log_variance).mean()
    total = (altitude_weight * altitude + phase_weight * phase
             + stop_weight * stop + uncertainty_weight * uncertainty)
    return ActiveAltitudeLosses(total, altitude, phase, stop, uncertainty)
