"""Optional RSRefSeg2-style semantic prior with calibrated late fusion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class SemanticPrior:
    logits: np.ndarray
    confidence: float
    source: str


class SemanticPriorProvider(Protocol):
    def get(self, map_name: str, instruction_id: str, grid_shape: tuple[int, int],
            altitude_level: int) -> Optional[SemanticPrior]: ...


class NullSemanticPrior:
    def get(self, map_name: str, instruction_id: str, grid_shape: tuple[int, int],
            altitude_level: int) -> Optional[SemanticPrior]:
        return None


class NpzSemanticPriorStore:
    """Read frozen offline priors; never invokes RSRefSeg2 inside a rollout."""

    def __init__(self, root: Path, confidence_floor: float = 0.25):
        self.root = Path(root)
        self.confidence_floor = confidence_floor

    def get(self, map_name: str, instruction_id: str, grid_shape: tuple[int, int],
            altitude_level: int) -> Optional[SemanticPrior]:
        path = self.root / map_name / f"{instruction_id}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as payload:
            logits = np.asarray(payload["logits"], dtype=np.float32).reshape(grid_shape)
            confidence = float(payload["confidence"])
        if not np.isfinite(logits).all() or confidence < self.confidence_floor:
            return None
        return SemanticPrior(logits=logits, confidence=min(confidence, 1.0),
                             source="rsrefseg2-offline")


def gated_semantic_logits(prior: Optional[SemanticPrior], expected_shape: tuple[int, int],
                          maximum_weight: float = 0.35) -> tuple[Optional[np.ndarray], float]:
    if prior is None:
        return None, 0.0
    logits = np.asarray(prior.logits, dtype=np.float32)
    if logits.shape != expected_shape:
        raise ValueError(f"semantic prior shape {logits.shape} != {expected_shape}")
    weight = float(np.clip(prior.confidence, 0.0, 1.0) * maximum_weight)
    # Centering prevents a provider-specific logit offset from changing fusion.
    return (logits - logits.mean()) * weight, float(prior.confidence)
