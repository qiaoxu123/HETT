"""Minimal persistent navigation state shared by future belief-update modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch


class CandidateStatus(str, Enum):
    UNKNOWN = "unknown"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    VISITED = "visited"
    CONFIRMED = "confirmed"


@dataclass
class BeliefCandidate:
    row: int
    col: int
    probability: float
    xy: tuple[float, float] | None = None
    status: CandidateStatus = CandidateStatus.UNKNOWN
    source: str = "belief_peak"
    nearby_landmarks: tuple[str, ...] = ()


@dataclass
class NavigationMemory:
    """Compact structured history; deliberately excludes raw RGB frame history."""

    trajectory_xy: list[tuple[float, float]] = field(default_factory=list)
    seen_landmarks: dict[str, float] = field(default_factory=dict)
    candidate_status: dict[int, CandidateStatus] = field(default_factory=dict)
    explored_mask: np.ndarray | None = None

    def update_pose(self, xy: tuple[float, float]) -> None:
        self.trajectory_xy.append((float(xy[0]), float(xy[1])))

    def mark_landmark_seen(self, name: str, confidence: float = 1.0) -> None:
        self.seen_landmarks[name] = max(float(confidence), self.seen_landmarks.get(name, 0.0))

    def set_candidate_status(self, candidate_index: int, status: CandidateStatus) -> None:
        self.candidate_status[int(candidate_index)] = CandidateStatus(status)

    def update_explored(self, observed_mask: np.ndarray) -> None:
        observed = np.asarray(observed_mask, dtype=bool)
        if self.explored_mask is None:
            self.explored_mask = observed.copy()
        else:
            if self.explored_mask.shape != observed.shape:
                raise ValueError("observed mask shape mismatch")
            self.explored_mask |= observed


def nms_topk_from_belief(
    probabilities: torch.Tensor | np.ndarray,
    *,
    top_k: int = 8,
    kernel_size: int = 3,
) -> list[BeliefCandidate]:
    """Greedy spatial NMS over a single 2-D normalized belief map."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    scores = torch.as_tensor(probabilities, dtype=torch.float32).detach().cpu().clone()
    if scores.ndim != 2:
        raise ValueError("probabilities must be 2-D")
    if not torch.isfinite(scores).all():
        raise ValueError("probabilities contain non-finite values")

    result: list[BeliefCandidate] = []
    radius = kernel_size // 2
    for _ in range(min(top_k, scores.numel())):
        flat_index = int(torch.argmax(scores).item())
        score = float(scores.flatten()[flat_index])
        if score < 0:
            break
        row, col = divmod(flat_index, scores.shape[1])
        result.append(BeliefCandidate(row=row, col=col, probability=score))

        r0, r1 = max(0, row - radius), min(scores.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(scores.shape[1], col + radius + 1)
        scores[r0:r1, c0:c1] = -1.0

    return result
