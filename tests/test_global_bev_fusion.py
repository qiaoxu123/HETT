import torch

from independent_nav.global_bev_fusion import (
    GlobalBEVFusion, decode_exact_coordinates, exact_location_loss,
)


def inputs(batch=2, steps=3, grid=8):
    prior = torch.full((batch, grid, grid), 1.0 / (grid * grid))
    landmark = torch.zeros(batch, grid, grid)
    relation = torch.randn(batch, 768)
    rgb = torch.rand(batch, steps, 3, grid, grid)
    visible = torch.ones(batch, steps, grid, grid, dtype=torch.bool)
    valid = torch.ones(batch, steps, dtype=torch.bool)
    return prior, landmark, relation, rgb, visible, valid


def test_forward_has_exact_location_outputs_and_no_target_argument():
    model = GlobalBEVFusion(grid_size=8, hidden=8)
    output = model(*inputs())
    assert output["belief_logits"].shape == (2, 3, 8, 8)
    assert output["offsets"].shape == (2, 3, 2, 8, 8)
    assert output["offsets"].abs().max() <= 0.5


def test_unseen_state_is_unchanged():
    model = GlobalBEVFusion(grid_size=8, hidden=8)
    prior, landmark, relation, rgb, _, _ = inputs(1, 1, 8)
    state = model.initial_state(prior, landmark, relation)
    invisible = torch.zeros(1, 8, 8, dtype=torch.bool)
    updated = model.fuse_step(state, rgb[:, 0], invisible)
    assert torch.equal(state, updated)


def test_exact_loss_rewards_correct_cell_and_offset():
    logits = torch.zeros(1, 1, 4, 4)
    offsets = torch.zeros(1, 1, 2, 4, 4)
    target = torch.tensor([[0.30, 0.70]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    wrong, _ = exact_location_loss(logits, offsets, target, valid)
    logits[0, 0, 2, 1] = 10.0
    offsets[0, 0, :, 2, 1] = torch.tensor([-0.30, 0.30])
    correct, _ = exact_location_loss(logits, offsets, target, valid)
    assert correct < wrong


def test_decode_applies_within_cell_offset():
    logits = torch.zeros(1, 1, 4, 4)
    logits[0, 0, 2, 1] = 5.0
    offsets = torch.zeros(1, 1, 2, 4, 4)
    offsets[0, 0, :, 2, 1] = torch.tensor([-0.25, 0.25])
    coordinate = decode_exact_coordinates(logits, offsets)
    assert torch.allclose(coordinate, torch.tensor([[0.3125, 0.6875]]))
