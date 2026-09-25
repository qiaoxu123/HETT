import torch

from independent_nav.trajectory_diffusion import (
    EndpointResidualUnet1D, endpoint_condition_features,
    lateral_residual_targets, residual_to_positions,
)


def test_endpoint_condition_features_are_explicit_and_finite():
    start = torch.tensor([[0.1, 0.2]])
    endpoint = torch.tensor([[[0.6, 0.2], [0.1, 0.7]]])
    minimum = torch.tensor([[5, 5]])
    feature = endpoint_condition_features(start, endpoint, minimum,
                                          torch.tensor([20]))
    assert feature.shape == (1, 2, 9)
    assert torch.isfinite(feature).all()
    assert torch.allclose(feature[0, :, :2], endpoint[0])


def test_direct_teacher_has_zero_lateral_residual():
    start = torch.tensor([[0.0, 0.0]])
    endpoint = torch.tensor([[[0.4, 0.0]]])
    positions = torch.zeros(1, 1, 21, 2)
    positions[0, 0, :5, 0] = torch.linspace(0, 0.4, 5)
    positions[0, 0, 5:, 0] = 0.4
    residual, minimum, extra, effective = lateral_residual_targets(
        positions, torch.tensor([[4]]), start, endpoint, step_scale=0.1)
    assert int(minimum[0, 0]) == 4
    assert int(extra[0, 0]) == 0
    assert int(effective[0, 0]) == 4
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_residual_path_keeps_exact_endpoint_and_bounded_steps():
    start = torch.tensor([[0.1, 0.1]])
    endpoint = torch.tensor([[[0.6, 0.1]]])
    residual = torch.zeros(1, 1, 20, 1)
    residual[0, 0, 1:4, 0] = torch.tensor([0.1, -0.1, 0.05])
    positions, length = residual_to_positions(
        residual, start, endpoint, torch.tensor([[5]]), step_scale=0.1)
    assert int(length[0, 0]) == 5
    assert torch.allclose(positions[0, 0, -1], endpoint[0, 0], atol=1e-6)
    step = torch.linalg.vector_norm(
        positions[0, 0, 1:6] - positions[0, 0, :5], dim=-1)
    assert bool(step.le(0.100001).all())


def test_endpoint_residual_model_outputs_noise_and_three_extra_classes():
    model = EndpointResidualUnet1D(base=8)
    batch = 2
    output = model(
        torch.zeros(batch, 20, 1), torch.zeros(batch, dtype=torch.long),
        torch.zeros(batch, 16, 16), torch.zeros(batch, 9, 768),
        torch.zeros(batch, 9, 12), torch.zeros(batch, 9, 2),
        torch.ones(batch, 9, dtype=torch.bool), torch.zeros(batch, 5),
        torch.zeros(batch, 9), torch.zeros(batch, dtype=torch.long))
    assert output["noise"].shape == (batch, 20, 1)
    assert output["extra_step_logits"].shape == (batch, 3)
