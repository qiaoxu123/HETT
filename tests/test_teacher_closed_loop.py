import torch

from scripts.evaluate_teacher_closed_loop import belief_prior


def test_belief_prior_uses_candidates_not_target():
    record = {"candidates": [[0.2, 0.2], [0.8, 0.8]], "target": [0.8, 0.8]}
    prior = belief_prior(record, grid_size=10)
    assert isinstance(prior, torch.Tensor)
    assert prior[2, 2] > prior[8, 8]
    assert prior.max() < 1.0
