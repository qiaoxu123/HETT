import inspect

import torch

from independent_nav.belief_transformer import (
    BeliefTransformer, TemporalBeliefTransformer, belief_loss, grid_centers, grid_landmark_geometry,
    rasterize_landmarks, target_mask,
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


def test_relative_geometry_uses_fixed_language_frame():
    cells = grid_centers(2).reshape(-1, 2)
    centers = torch.tensor([[[0.25, 0.25]]])
    boundaries = centers[:, :, None].expand(-1, -1, 16, -1).clone()
    geometry = grid_landmark_geometry(cells, centers, boundaries)
    # The bottom-left grid cell coincides with the landmark; the top-right cell
    # has positive map dy, which becomes positive language-frame x after R270.
    assert torch.allclose(geometry[0, 0, 0, :2], torch.zeros(2), atol=1e-6)
    assert geometry[0, 3, 0, 0] > 0
    assert geometry[0, 3, 0, 1] < 0


def test_temporal_evidence_is_visibility_gated():
    inputs = synthetic_inputs(batch=2, landmarks=3, grid=8)
    static = BeliefTransformer(
        grid_size=8, hidden=32, layers=1, heads=4, dropout=0.0)
    model = TemporalBeliefTransformer(static, hidden=16).eval()
    semantic = torch.rand(2, 3, 5, 8, 8)
    visibility = torch.ones(2, 3, 8, 8, dtype=torch.bool)
    visibility[:, :, :2] = False
    output = model(inputs, semantic, visibility)
    assert output["belief_logits"].shape == (2, 3, 8, 8)
    assert output["raw_evidence"].shape == (2, 3, 8, 8)
    assert output["target_type_logits"].shape == (2, 4)
    assert torch.equal(
        output["applied_evidence"][:, :, :2],
        torch.zeros_like(output["applied_evidence"][:, :, :2]))
    assert torch.isfinite(output["belief_logits"]).all()
