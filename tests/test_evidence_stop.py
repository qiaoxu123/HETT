import inspect

import numpy as np
import torch

from independent_nav.evidence_stop import (
    EvidenceStopHead, center_semantic_statistics, first_stop_index,
    local_belief_statistics, stop_features,
)


def test_stop_feature_builder_has_no_reference_target_input():
    assert "target" not in inspect.signature(stop_features).parameters
    value = stop_features(
        (0.1, 0.2), (0.2, 0.2), (0.8, 0.8), [0.7, 0.3], [0.9, 0.1],
        (0.8, 0.7, 0.9), (0.6, 0.2), True, 12)
    assert value.shape == (14,)
    assert np.isfinite(value).all()


def test_center_statistics_respond_to_requested_target_class():
    probability = np.zeros((3, 9, 9), dtype=np.float32)
    probability[0] = 0.1
    probability[1] = 0.8
    probability[2] = 0.1
    assert center_semantic_statistics(probability, 1, 3)[0] > 0.7
    assert center_semantic_statistics(probability, 2, 3)[0] < 0.2


def test_stop_head_returns_one_logit_per_state():
    assert EvidenceStopHead()(torch.randn(5, 14)).shape == (5,)


def test_first_stop_requires_eligibility_and_falls_back_to_final_state():
    assert first_stop_index([0.9, 0.8, 0.7], [False, True, True], 0.75) == 1
    assert first_stop_index([0.1, 0.2], [True, True], 0.9) == 1


def test_local_belief_statistics_peak_at_high_logit_cell():
    logits = np.zeros((5, 5), dtype=np.float32)
    logits[2, 3] = 5
    at_peak = local_belief_statistics(logits, (0.7, 0.5))
    away = local_belief_statistics(logits, (0.1, 0.1))
    assert at_peak[0] > 0.99
    assert at_peak[1] > away[1]
