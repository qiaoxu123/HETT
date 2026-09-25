"""Learned evidence stopping without reference-target inputs at inference."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "distance_to_selected_mode",
    "distance_to_alternate_mode",
    "selected_posterior",
    "posterior_margin",
    "selected_rgb_evidence",
    "rgb_evidence_margin",
    "center_target_soft_probability",
    "center_target_hard_fraction",
    "center_semantic_confidence",
    "current_belief_relative_peak",
    "current_belief_local_mass",
    "evidence_acquired",
    "remaining_budget_fraction",
    "mode_separation",
)


class EvidenceStopHead(nn.Module):
    def __init__(self, input_dim=len(FEATURE_NAMES), hidden=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def _circle_mask(height, width, radius_pixels):
    yy, xx = np.ogrid[:height, :width]
    return ((xx - (width - 1) / 2) ** 2 + (yy - (height - 1) / 2) ** 2
            <= radius_pixels ** 2)


def center_semantic_statistics(probability, target_class, radius_pixels):
    """Return inference-visible target evidence around the drone position."""
    probability = np.asarray(probability, dtype=np.float64)
    mask = _circle_mask(probability.shape[1], probability.shape[2], radius_pixels)
    target = probability[int(target_class)][mask]
    prediction = probability.argmax(0)[mask]
    confidence = probability.max(0)[mask]
    return (float(target.mean()), float(np.mean(prediction == int(target_class))),
            float(confidence.mean()))


def stop_features(current_xy, selected_xy, alternate_xy, posterior, rgb_evidence,
                  center_statistics, belief_statistics, evidence_acquired,
                  remaining_actions,
                  max_actions=20, map_size_m=410.0):
    """Build stop features; deliberately has no target-coordinate argument."""
    current = np.asarray(current_xy, dtype=np.float64)
    selected = np.asarray(selected_xy, dtype=np.float64)
    alternate = np.asarray(alternate_xy, dtype=np.float64)
    posterior = np.asarray(posterior, dtype=np.float64)
    posterior = posterior / posterior.sum()
    evidence = np.asarray(rgb_evidence, dtype=np.float64)
    chosen = int(posterior.argmax())
    other = 1 - chosen
    return np.asarray([
        np.linalg.norm(current - selected) * map_size_m / 100.0,
        np.linalg.norm(current - alternate) * map_size_m / 100.0,
        posterior[chosen], posterior[chosen] - posterior[other],
        evidence[chosen], evidence[chosen] - evidence[other],
        center_statistics[0], center_statistics[1], center_statistics[2],
        belief_statistics[0], belief_statistics[1],
        float(evidence_acquired), remaining_actions / max_actions,
        np.linalg.norm(selected - alternate) * map_size_m / 100.0,
    ], dtype=np.float32)


def local_belief_statistics(belief_logits, current_xy, radius_cells=1):
    """Measure language/map belief support at the current inference pose."""
    logits = np.asarray(belief_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("belief logits must be a square map")
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    size = logits.shape[0]
    x = min(size - 1, max(0, int(float(current_xy[0]) * size)))
    y = min(size - 1, max(0, int(float(current_xy[1]) * size)))
    y0, y1 = max(0, y - radius_cells), min(size, y + radius_cells + 1)
    x0, x1 = max(0, x - radius_cells), min(size, x + radius_cells + 1)
    relative_peak = probability[y, x] / max(float(probability.max()), 1e-12)
    local_mass = probability[y0:y1, x0:x1].sum()
    return float(relative_peak), float(local_mass)


def first_stop_index(probabilities, eligible, threshold):
    """Select the first eligible learned stop, otherwise preserve final stop."""
    for index, (probability, allowed) in enumerate(zip(probabilities, eligible)):
        if allowed and probability >= threshold:
            return index
    return len(probabilities) - 1
