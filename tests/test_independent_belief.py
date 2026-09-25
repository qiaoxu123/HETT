import numpy as np

import torch

from independent_nav.belief import PositiveEvidenceBelief, VisibilityGatedBeliefMap


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


def test_signed_evidence_can_raise_and_lower_visible_cells():
    belief = VisibilityGatedBeliefMap(torch.full((2, 2), 0.5))
    visible = torch.tensor([[True, True], [False, True]])
    result = belief.update(
        torch.tensor([[2.0, -2.0], [-4.0, 0.0]]), visible,
        evidence_id="frame-1")
    assert result[0, 0] > 0.5
    assert result[0, 1] < 0.5
    assert result[1, 0] == 0.5


def test_occluded_negative_and_low_quality_evidence_are_ignored():
    belief = VisibilityGatedBeliefMap(torch.full((2,), 0.6), quality_floor=0.3)
    initial = belief.probability.clone()
    belief.update(torch.tensor([-5.0, -5.0]), torch.tensor([False, True]),
                  quality=torch.tensor([1.0, 0.2]), evidence_id=1)
    assert torch.equal(belief.probability, initial)
    assert belief.observations == 0


def test_duplicate_frame_cannot_accumulate_forever():
    belief = VisibilityGatedBeliefMap(torch.tensor([0.5]))
    first = belief.update(torch.tensor([1.0]), torch.tensor([True]),
                          evidence_id="same").clone()
    second = belief.update(torch.tensor([1.0]), torch.tensor([True]),
                           evidence_id="same")
    assert torch.equal(first, second)
    assert belief.observations == 1


def test_uncertainty_controls_second_view_trigger():
    diffuse = VisibilityGatedBeliefMap(torch.full((4,), 0.5))
    concentrated = VisibilityGatedBeliefMap(torch.tensor([0.98, 0.01, 0.01, 0.01]))
    assert diffuse.needs_another_view(entropy_threshold=0.8)
    assert not concentrated.needs_another_view(
        entropy_threshold=0.8, peak_threshold=0.9)
