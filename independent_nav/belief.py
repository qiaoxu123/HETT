"""Small target-free temporal belief primitives for independent navigation."""

from dataclasses import dataclass

import torch


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


class VisibilityGatedBeliefMap:
    """Bayesian-style signed evidence accumulation in log-odds space.

    The caller supplies an observation log-likelihood ratio. Positive values
    support a cell and negative values reject it. Negative evidence is applied
    only where ``observable`` is true; occluded or invalid pixels preserve the
    previous belief. Reused evidence IDs are ignored to prevent counting the
    same frame repeatedly.
    """

    def __init__(self, prior_probability: torch.Tensor, quality_floor: float = 0.25,
                 logit_limit: float = 10.0):
        prior = prior_probability.float().clamp(1e-5, 1.0 - 1e-5)
        self.log_odds = torch.logit(prior)
        self.quality_floor = float(quality_floor)
        self.logit_limit = float(logit_limit)
        self.observations = 0
        self._evidence_ids = set()

    @property
    def probability(self) -> torch.Tensor:
        return self.log_odds.sigmoid()

    def update(self, log_likelihood_ratio: torch.Tensor,
               observable: torch.Tensor, quality=1.0,
               evidence_id=None) -> torch.Tensor:
        if evidence_id is not None and evidence_id in self._evidence_ids:
            return self.probability
        evidence = torch.as_tensor(
            log_likelihood_ratio, dtype=self.log_odds.dtype,
            device=self.log_odds.device)
        visible = torch.as_tensor(
            observable, dtype=torch.bool, device=self.log_odds.device)
        quality_tensor = torch.as_tensor(
            quality, dtype=self.log_odds.dtype, device=self.log_odds.device)
        if quality_tensor.ndim == 0:
            quality_tensor = quality_tensor.expand_as(self.log_odds)
        reliable = visible & quality_tensor.ge(self.quality_floor)
        weighted = evidence * quality_tensor.clamp(0.0, 1.0)
        self.log_odds = torch.where(
            reliable,
            (self.log_odds + weighted).clamp(-self.logit_limit, self.logit_limit),
            self.log_odds)
        if evidence_id is not None:
            self._evidence_ids.add(evidence_id)
        if reliable.any():
            self.observations += 1
        return self.probability

    def normalized(self) -> torch.Tensor:
        probability = self.probability
        return probability / probability.sum().clamp_min(1e-8)

    def entropy(self) -> float:
        probability = self.normalized().flatten()
        return float(-(probability * probability.clamp_min(1e-8).log()).sum())

    def needs_another_view(self, entropy_threshold: float,
                           peak_threshold: float = 0.5) -> bool:
        return (self.entropy() > float(entropy_threshold)
                or float(self.probability.max()) < float(peak_threshold))
