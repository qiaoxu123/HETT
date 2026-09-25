import inspect

import torch

from independent_nav.belief_transformer import (
    BeliefTransformer, belief_loss, rasterize_landmarks, target_mask,
)


def synthetic_inputs(batch=2, landmarks=3, grid=8):
    torch.manual_seed(2)
    centers = torch.rand(batch, landmarks, 2)
    half = torch.full((batch, landmarks, 2), 0.04)
    corners = torch.stack((
        centers - half,
        centers + torch.stack((half[..., 0], -half[..., 1]), -1),
        centers + half,
        centers + torch.stack((-half[..., 0], half[..., 1]), -1),
    ), dim=2)
    boundaries = corners.repeat_interleave(4, dim=2)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    masks = rasterize_landmarks(boundaries, valid, grid)
    union = masks.any(1).float()
    bev = torch.stack((union, torch.zeros_like(union)), dim=1)
    return (torch.randn(batch, landmarks, 768),
            torch.randn(batch, landmarks, 12), centers, boundaries,
            valid, masks, bev)


def test_model_forward_has_no_target_coordinate_argument():
    names = inspect.signature(BeliefTransformer.forward).parameters
    assert "target" not in names and "target_xy" not in names


def test_forward_shape_and_finite_loss():
    inputs = synthetic_inputs()
    model = BeliefTransformer(grid_size=8, hidden=32, layers=1, heads=4).eval()
    logits = model(*inputs)
    labels = target_mask(torch.tensor([[0.2, 0.3], [0.7, 0.8]]), 8)
    loss = belief_loss(logits, labels)
    assert logits.shape == (2, 8, 8)
    assert torch.isfinite(loss)


def test_landmark_permutation_invariance():
    inputs = list(synthetic_inputs())
    model = BeliefTransformer(
        grid_size=8, hidden=32, layers=1, heads=4, dropout=0.0).eval()
    order = torch.tensor([2, 0, 1])
    with torch.inference_mode():
        expected = model(*inputs)
        permuted = [value[:, order] if index < 6 else value
                    for index, value in enumerate(inputs)]
        actual = model(*permuted)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_target_mask_is_metric_radius():
    target = torch.tensor([[0.5, 0.5]])
    mask = target_mask(target, grid_size=20, radius_m=20.0)
    assert mask.any()
    assert not mask.all()
