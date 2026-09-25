import torch

from scripts.evaluate_rgb_relation_signatures import candidate_masks, top_fraction


def test_candidate_masks_have_metric_local_support():
    masks = candidate_masks([[0.5, 0.5], [0.1, 0.1], [0.9, 0.9]])
    assert masks.shape == (3, 20, 20)
    assert masks[0].any() and not masks[0].all()


def test_top_fraction_ignores_outside_values():
    value = torch.zeros(2, 2)
    value[0, 0] = 0.8
    value[1, 1] = 1.0
    mask = torch.tensor([[True, False], [False, False]])
    assert top_fraction(value, mask) == torch.tensor(0.8).item()
