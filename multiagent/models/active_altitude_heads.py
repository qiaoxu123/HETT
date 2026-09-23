"""Heads for discrete active sensing, phase prediction and semantic stopping."""

import torch
from torch import nn


class ActiveAltitudeHeads(nn.Module):
    """Small heads intended to consume the fused HETT motion token.

    Returned logits remain raw: losses, temperature calibration and controller
    thresholds live outside the network so every component can be ablated.
    """

    def __init__(self, hidden_size: int = 768, dropout: float = 0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.altitude_action = nn.Linear(128, 3)
        self.phase = nn.Linear(128, 4)
        self.stop = nn.Linear(128, 1)
        self.goal_log_variance = nn.Linear(128, 1)

    def forward(self, fused_motion_token: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.shared(fused_motion_token)
        return {
            "altitude_action_logits": self.altitude_action(hidden),
            "phase_logits": self.phase(hidden),
            "stop_logit": self.stop(hidden),
            "goal_log_variance": self.goal_log_variance(hidden).clamp(-6.0, 4.0),
        }
