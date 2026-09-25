import torch

from scripts.train_temporal_belief_transformer import evidence_loss


def test_evidence_loss_ignores_invisible_cells():
    labels = torch.tensor([[[True, False], [False, False]]])
    visible = torch.tensor([[[[True, False], [False, False]]]])
    good = torch.tensor([[[[5.0, -5.0], [-5.0, -5.0]]]])
    altered_outside = good.clone()
    altered_outside[~visible] = 50.0
    assert torch.allclose(
        evidence_loss(good, labels, visible),
        evidence_loss(altered_outside, labels, visible))


def test_evidence_loss_rewards_correct_signed_evidence():
    labels = torch.tensor([[[True, False], [False, False]]])
    visible = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    good = torch.tensor([[[[5.0, -5.0], [-5.0, -5.0]]]])
    bad = -good
    assert evidence_loss(good, labels, visible) < evidence_loss(bad, labels, visible)


def test_fully_padded_timestep_contributes_no_evidence_loss():
    labels = torch.tensor([[[True, False], [False, False]]])
    raw = torch.randn(1, 1, 2, 2)
    invisible = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    assert evidence_loss(raw, labels, invisible) == 0
