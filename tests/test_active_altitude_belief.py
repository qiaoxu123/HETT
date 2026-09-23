import numpy as np

from multiagent.active_altitude.belief import MultiScaleBelief
from multiagent.active_altitude.semantics import SemanticPrior, gated_semantic_logits


def test_high_altitude_evidence_is_broader_than_low_altitude_evidence():
    logits = np.zeros((5, 5), dtype=np.float32)
    logits[2, 2] = 10
    low = MultiScaleBelief((5, 5))
    high = MultiScaleBelief((5, 5))
    low_probs = low.update(logits, altitude_m=30)
    high_probs = high.update(logits, altitude_m=120)
    assert low_probs[2, 2] > high_probs[2, 2]


def test_rsrefseg2_prior_is_confidence_gated_and_centered():
    prior = SemanticPrior(np.arange(25).reshape(5, 5), 0.8, "test")
    logits, confidence = gated_semantic_logits(prior, (5, 5), maximum_weight=0.25)
    assert confidence == 0.8
    assert np.isclose(logits.mean(), 0.0)
    assert np.isclose(logits[-1, -1], (24 - 12) * 0.2)
