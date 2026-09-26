import inspect
from types import SimpleNamespace

import torch

from multiagent.models.multilandmark_belief import (
    MultiLandmarkBeliefHead,
    make_grid_centers,
    target_cell_ids,
    target_region_mask,
)
from multiagent.maps.landmark_map import LandmarkMap


def make_inputs(batch=2, landmarks=4, dim=32):
    torch.manual_seed(7)
    context = torch.randn(batch, dim)
    centers = torch.rand(batch, landmarks, 2)
    names = torch.randn(batch, landmarks, dim)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    return context, centers, names, valid


def test_output_shapes_bounds_and_gradients():
    model = MultiLandmarkBeliefHead(d_model=32, hidden_dim=16, grid_size=7)
    inputs = make_inputs()
    previous = torch.randn(2, 49)
    logits, offsets, goals, context = model(*inputs, previous_belief=previous)
    assert logits.shape == (2, 49)
    assert offsets.shape == (2, 49, 2)
    assert goals.shape == (2, 2)
    assert context.shape == (2, 32)
    assert torch.all((goals >= 0) & (goals <= 1))
    (logits.mean() + offsets.mean() + goals.mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_landmark_order_is_permutation_invariant():
    model = MultiLandmarkBeliefHead(d_model=32, hidden_dim=16, grid_size=7).eval()
    context, centers, names, valid = make_inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        original = model(context, centers, names, valid)
        permuted = model(
            context,
            centers[:, permutation],
            names[:, permutation],
            valid[:, permutation],
        )
    for original_tensor, permuted_tensor in zip(original, permuted):
        torch.testing.assert_close(original_tensor, permuted_tensor, rtol=1e-5, atol=1e-6)


def test_invalid_landmarks_are_ignored():
    model = MultiLandmarkBeliefHead(d_model=32, hidden_dim=16, grid_size=7).eval()
    context, centers, names, valid = make_inputs()
    changed_centers = centers.clone()
    changed_names = names.clone()
    changed_centers[~valid] = 1000
    changed_names[~valid] = 1000
    with torch.no_grad():
        original = model(context, centers, names, valid)
        changed = model(context, changed_centers, changed_names, valid)
    for original_tensor, changed_tensor in zip(original, changed):
        torch.testing.assert_close(original_tensor, changed_tensor, rtol=1e-5, atol=1e-6)


def test_previous_belief_is_accumulated():
    model = MultiLandmarkBeliefHead(d_model=32, hidden_dim=16, grid_size=7).eval()
    inputs = make_inputs()
    previous = torch.zeros(2, 49)
    previous[:, 3] = 4.0
    with torch.no_grad():
        without_history = model(*inputs)[0]
        with_history = model(*inputs, previous_belief=previous)[0]
    assert torch.all(with_history[:, 3] > without_history[:, 3])


def test_belief_is_normalized_and_can_correct_an_old_peak():
    model = MultiLandmarkBeliefHead(d_model=32, hidden_dim=16, grid_size=7).eval()
    inputs = make_inputs()
    previous = torch.full((2, 49), -8.0)
    previous[:, 3] = 0.0
    with torch.no_grad():
        updated = previous
        for _ in range(12):
            updated = model(*inputs, previous_belief=updated)[0]
    torch.testing.assert_close(
        torch.logsumexp(updated, dim=-1), torch.zeros(2), atol=1e-5, rtol=1e-5
    )
    assert torch.isfinite(updated).all()


def test_all_invalid_landmarks_use_context_fallback():
    model = MultiLandmarkBeliefHead(d_model=32, hidden_dim=16, grid_size=7).eval()
    context, centers, names, valid = make_inputs()
    valid.zero_()
    with torch.no_grad():
        outputs = model(context, centers, names, valid)
    assert all(torch.isfinite(output).all() for output in outputs)


def test_zero_confidence_ignores_matched_landmark_geometry():
    model = MultiLandmarkBeliefHead(d_model=32, hidden_dim=16, grid_size=7).eval()
    context, centers, names, valid = make_inputs()
    confidence = torch.zeros_like(valid, dtype=torch.float32)
    changed_centers = centers + 1000
    with torch.no_grad():
        original = model(context, centers, names, valid, landmark_confidence=confidence)
        changed = model(context, changed_centers, names, valid, landmark_confidence=confidence)
    torch.testing.assert_close(original[0], changed[0], rtol=1e-5, atol=1e-6)


def test_target_labels_use_metric_success_region():
    goals = torch.tensor([[0.5, 0.5], [0.99, 0.99]])
    grid = make_grid_centers(41)
    ids = target_cell_ids(goals, 41)
    mask = target_region_mask(goals, grid, radius_m=20.0, map_meters=410.0)
    assert ids.shape == (2,)
    assert mask.shape == (2, 41 * 41)
    assert mask[0, ids[0]]
    assert mask[1].any()


def test_inference_head_cannot_receive_target_truth():
    parameters = inspect.signature(MultiLandmarkBeliefHead.forward).parameters
    forbidden = {'target', 'goal', 'gt_goal', 'target_position', 'normalized_goal'}
    assert forbidden.isdisjoint(parameters)


def test_landmark_fuzzy_match_can_reject_unrelated_text():
    old_cache = LandmarkMap._landmarks_cache
    LandmarkMap._landmarks_cache = {
        'map': {
            1: SimpleNamespace(name='Central Library'),
            2: SimpleNamespace(name='King Street'),
        }
    }
    try:
        exact = LandmarkMap._search_landmarks_by_name(
            'map', ['central library'], min_similarity=0.7
        )
        rejected = LandmarkMap._search_landmarks_by_name(
            'map', ['multi-colored apartment block beside a car park'],
            min_similarity=0.7,
        )
    finally:
        LandmarkMap._landmarks_cache = old_cache
    assert exact[0][1].name == 'Central Library'
    assert rejected == []
