"""Small target-free temporal belief primitives for independent navigation."""

from dataclasses import dataclass


@dataclass
class PositiveEvidenceBelief:
    """Accumulate repeatable positive observations without treating occlusion as absence."""

    evidence_floor: float = 0.1
    confidence: float = 0.0
    observations: int = 0

    def update(self, score: float, observable: bool = True) -> float:
        if not observable:
            return self.confidence
        calibrated = min(1.0, max(0.0, (float(score) - self.evidence_floor)
                                      / max(1.0 - self.evidence_floor, 1e-6)))
        self.confidence = 1.0 - (1.0 - self.confidence) * (1.0 - calibrated)
        self.observations += 1
        return self.confidence
