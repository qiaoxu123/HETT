"""Altitude-aware multi-observation goal belief fusion."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BeliefDiagnostics:
    normalized_entropy: float
    top2_margin: float
    peak_probability: float
    observations: int


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / np.clip(values.sum(), 1e-12, None)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(2.5 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(axis ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _blur(grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return grid
    kernel = _gaussian_kernel(sigma)
    radius = len(kernel) // 2
    padded = np.pad(grid, ((radius, radius), (0, 0)), mode="edge")
    vertical = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 0, padded)
    padded = np.pad(vertical, ((0, 0), (radius, radius)), mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)


class MultiScaleBelief:
    """Fuse observations in log space while respecting altitude resolution.

    High-altitude evidence is deliberately blurred and down-weighted.  An
    optional semantic prior is expected to be confidence-gated before this call;
    it can therefore be ablated without changing the navigation model or controller.
    """

    def __init__(self, grid_shape: tuple[int, int], reference_altitude_m: float = 30.0,
                 semantic_weight: float = 1.0):
        self.grid_shape = grid_shape
        self.reference_altitude_m = reference_altitude_m
        self.semantic_weight = semantic_weight
        self.log_belief = np.zeros(grid_shape, dtype=np.float64)
        self.observations = 0

    def reset(self):
        self.log_belief.fill(0.0)
        self.observations = 0

    def update(self, spatial_logits: Sequence[float], altitude_m: float,
               semantic_logits: Optional[Sequence[float]] = None,
               reliability: float = 1.0) -> np.ndarray:
        logits = np.asarray(spatial_logits, dtype=np.float64).reshape(self.grid_shape)
        scale = max(float(altitude_m) / self.reference_altitude_m, 1.0)
        probabilities = _softmax(logits.reshape(-1)).reshape(self.grid_shape)
        probabilities = _blur(probabilities, sigma=0.45 * (scale - 1.0))
        probabilities /= np.clip(probabilities.sum(), 1e-12, None)

        # Fine observations receive more weight; no observation is allowed to
        # erase all prior evidence in one step.
        visual_weight = float(np.clip(reliability, 0.0, 1.0)) / np.sqrt(scale)
        self.log_belief += visual_weight * np.log(np.clip(probabilities, 1e-8, None))
        if semantic_logits is not None:
            semantic = _softmax(np.asarray(semantic_logits, dtype=np.float64).reshape(-1))
            semantic = semantic.reshape(self.grid_shape)
            self.log_belief += self.semantic_weight * np.log(np.clip(semantic, 1e-8, None))
        self.log_belief -= self.log_belief.max()
        self.observations += 1
        return self.probabilities()

    def probabilities(self) -> np.ndarray:
        return _softmax(self.log_belief.reshape(-1)).reshape(self.grid_shape)

    def diagnostics(self) -> BeliefDiagnostics:
        probs = self.probabilities().reshape(-1)
        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        entropy /= np.log(probs.size)
        top2 = np.partition(probs, -2)[-2:]
        return BeliefDiagnostics(
            normalized_entropy=float(np.clip(entropy, 0.0, 1.0)),
            top2_margin=float(top2.max() - top2.min()),
            peak_probability=float(top2.max()),
            observations=self.observations,
        )
