"""Temporal target hypotheses built only from current/past model predictions."""

import torch


class TemporalHypothesisTracker:
    def __init__(self, num_candidates, top_k=3, decay=0.8):
        if num_candidates < 2:
            raise ValueError('num_candidates must be at least two')
        if not 1 <= top_k <= num_candidates:
            raise ValueError('top_k must be within candidate count')
        if not 0 <= decay < 1:
            raise ValueError('decay must be in [0, 1)')
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.decay = decay
        self.evidence = None

    def update(self, logits, candidate_positions):
        """Update beliefs and return the current MAP position plus diagnostics."""
        logits = logits.detach().float().reshape(-1).cpu()
        positions = candidate_positions.detach().float().reshape(self.num_candidates, 2).cpu()
        if logits.numel() != self.num_candidates:
            raise ValueError('logit count does not match tracker candidates')
        if not torch.isfinite(logits).all() or not torch.isfinite(positions).all():
            raise ValueError('hypothesis inputs must be finite')

        probabilities = torch.softmax(logits, dim=0)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        max_entropy = torch.tensor(float(self.num_candidates)).log()
        confidence = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
        centered_log_evidence = torch.log_softmax(logits, dim=0) + max_entropy
        update = confidence * centered_log_evidence
        self.evidence = update if self.evidence is None else self.decay * self.evidence + update

        posterior = torch.softmax(self.evidence, dim=0)
        top_probability, top_index = posterior.topk(self.top_k)
        best = int(top_index[0])
        return positions[best], {
            'indices': top_index.tolist(),
            'probabilities': top_probability.tolist(),
            'confidence': float(confidence),
        }
