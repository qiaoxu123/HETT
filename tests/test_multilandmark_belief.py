import inspect

import torch

from multiagent.models.multilandmark_belief import (
    MultiLandmarkBeliefHead,
    make_grid_centers,
    target_cell_ids,
    target_region_mask,
)


def make_inputs(batch=2, landmarks=4, dim=32):
    torch.manual_seed(7)
    context = torch.randn(batch, dim)
    centers = torch.rand(batch, landmarks, 2)
    boundaries = centers[:, :, None, :] + 0.02 * torch.randn(batch, landmarks, 16, 2)
    names = torch.randn(batch, landmarks, dim)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    return context, centers, boundaries, names, valid


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
    context, centers, boundaries, names, valid = make_inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        original = model(context, centers, boundaries, names, valid)
        permuted = model(
            context,
            centers[:, permutation],
            boundaries[:, permutation],
            names[:, permutation],
            valid[:, permutation],
        )
    for original_tensor, permuted_tensor in zip(original, permuted):
        torch.testing.assert_close(original_tensor, permuted_tensor, rtol=1e-5, atol=1e-6)


def test_invalid_landmarks_are_ignored():
    model = MultiLandmarkBeliefHead(d_model=32, hidden_dim=16, grid_size=7).eval()
    context, centers, boundaries, names, valid = make_inputs()
    changed_centers = centers.clone()
    changed_boundaries = boundaries.clone()
    changed_names = names.clone()
    changed_centers[~valid] = 1000
    changed_names[~valid] = 1000
    changed_boundaries[~valid] = 1000
    with torch.no_grad():
        original = model(context, centers, boundaries, names, valid)
        changed = model(context, changed_centers, changed_boundaries, changed_names, valid)
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


def test_valid_landmark_boundary_changes_spatial_logits():
    model = MultiLandmarkBeliefHead(d_model=32, hidden_dim=16, grid_size=7).eval()
    context, centers, boundaries, names, valid = make_inputs()
    expanded = boundaries.clone()
    expanded[:, 0] += torch.tensor([0.15, -0.10])
    with torch.no_grad():
        original = model(context, centers, boundaries, names, valid)[0]
        changed = model(context, centers, expanded, names, valid)[0]
    assert not torch.allclose(original, changed)


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


def test_camera_pose_prior_peaks_near_current_position():
    model = MultiLandmarkBeliefHead(d_model=32, hidden_dim=16, grid_size=41).eval()
    context, centers, boundaries, names, valid = make_inputs()
    current_positions = torch.tensor([[0.20, 0.30], [0.75, 0.60]])
    with torch.no_grad():
        _, _, goals, _ = model(
            context,
            centers,
            boundaries,
            names,
            valid,
            current_positions=current_positions,
        )
    errors_m = torch.linalg.vector_norm(goals - current_positions, dim=-1) * 410.0
    assert torch.all(errors_m < 20.0)
