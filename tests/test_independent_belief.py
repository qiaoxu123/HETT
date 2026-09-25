import numpy as np

from independent_nav.belief import PositiveEvidenceBelief


def test_positive_evidence_accumulates_without_false_negative_updates():
    belief = PositiveEvidenceBelief(evidence_floor=0.1)
    first = belief.update(0.55)
    second = belief.update(0.55)
    assert second > first > 0
    assert belief.observations == 2
    assert np.isclose(belief.update(0.0), second)


def test_unobservable_frame_does_not_change_belief():
    belief = PositiveEvidenceBelief()
    belief.update(0.8)
    before = belief.confidence
    assert belief.update(1.0, observable=False) == before
    assert belief.observations == 1
