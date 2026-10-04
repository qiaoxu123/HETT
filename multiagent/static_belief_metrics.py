"""Training targets and offline diagnostics for the static initial belief."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from multiagent.navigation_state import nms_topk_from_belief


def gaussian_field_target(
    height: int,
    width: int,
    goal_row_col: tuple[float, float],
    *,
    sigma_cells: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if sigma_cells <= 0:
        raise ValueError("sigma_cells must be positive")
    rows = torch.arange(height, dtype=torch.float32, device=device)
    cols = torch.arange(width, dtype=torch.float32, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    gr, gc = goal_row_col
    target = torch.exp(-((rr - gr) ** 2 + (cc - gc) ** 2) / (2 * sigma_cells ** 2))
    return target / target.sum().clamp_min(torch.finfo(target.dtype).eps)


def soft_field_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("logits and target must have the same shape")
    log_probs = F.log_softmax(logits.flatten(1), dim=-1)
    normalized = target.flatten(1)
    normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(normalized * log_probs).sum(dim=-1).mean()


def candidate_coverage(
    probabilities: torch.Tensor | np.ndarray,
    goal_row_col: tuple[float, float],
    *,
    ks: Iterable[int] = (1, 4, 8, 16),
    radii_cells: Iterable[float] = (1.0, 2.0, 4.0),
    nms_kernel: int = 3,
) -> dict[str, float]:
    ks = tuple(ks)
    radii_cells = tuple(radii_cells)
    if not ks:
        raise ValueError("ks must be non-empty")
    max_k = max(ks)
    candidates = nms_topk_from_belief(probabilities, top_k=max_k, kernel_size=nms_kernel)
    goal = np.asarray(goal_row_col, dtype=np.float32)
    distances = np.asarray(
        [math.hypot(c.row - goal[0], c.col - goal[1]) for c in candidates],
        dtype=np.float32,
    )

    metrics: dict[str, float] = {}
    for k in ks:
        prefix = distances[: min(k, len(distances))]
        metrics[f"oracle_distance@{k}_cells"] = float(prefix.min()) if len(prefix) else float("inf")
        for radius in radii_cells:
            metrics[f"recall@{k}/{radius:g}cells"] = float(bool(len(prefix)) and np.any(prefix <= radius))
    return metrics


def belief_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).eps)
    return -(probabilities * probabilities.log()).flatten(1).sum(dim=-1)
