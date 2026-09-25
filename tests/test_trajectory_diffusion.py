import torch

from independent_nav.trajectory_diffusion import (
    ConditionalUnet1D, bounded_endpoint_projection, cosine_noise_schedule,
    ddim_sample,
    make_ddim_scheduler, make_ddpm_scheduler,
)


def condition(batch=2, landmarks=3, grid=8):
    return (
        torch.randn(batch, grid, grid),
        torch.randn(batch, landmarks, 768),
        torch.randn(batch, landmarks, 12),
        torch.rand(batch, landmarks, 2),
        torch.ones(batch, landmarks, dtype=torch.bool),
        torch.randn(batch, 5),
    )


def test_conditional_unet_shapes_and_target_free_signature():
    model = ConditionalUnet1D(base=16, horizon=20).eval()
    values = condition()
    with torch.inference_mode():
        output = model(
            torch.randn(2, 20, 4), torch.tensor([1, 7]), *values,
            trajectory_slot=torch.tensor([0, 1]))
    assert output["noise"].shape == (2, 20, 4)
    assert output["length_logits"].shape == (2, 21)


def test_ddim_generates_multiple_budgeted_length_hypotheses():
    model = ConditionalUnet1D(base=16, horizon=20).eval()
    values = condition(batch=1)
    actions, length = ddim_sample(
        model, values, samples=3, diffusion_steps=8, inference_steps=4)
    assert actions.shape == (1, 3, 20, 4)
    assert length.shape == (1, 3)
    assert length.min() >= 0 and length.max() <= 20


def test_cosine_schedule_ends_near_pure_noise():
    _, _, alpha_bar = cosine_noise_schedule(100)
    assert alpha_bar[-1] < 1e-4


def test_official_diffusers_schedulers_share_training_noise():
    ddpm = make_ddpm_scheduler(100)
    ddim = make_ddim_scheduler(100)
    assert torch.allclose(ddpm.alphas_cumprod, ddim.alphas_cumprod)


def test_endpoint_projection_is_exact_and_step_bounded():
    actions = torch.randn(1, 2, 20, 4)
    starts = torch.tensor([[0.1, 0.2]])
    endpoints = torch.tensor([[[0.8, 0.2], [0.1, 0.9]]])
    positions, lengths = bounded_endpoint_projection(
        actions, torch.tensor([[3, 4]]), starts, endpoints, 0.1)
    selected = torch.stack((
        positions[0, 0, lengths[0, 0]],
        positions[0, 1, lengths[0, 1]],
    ))
    assert torch.allclose(selected, endpoints[0], atol=1e-5)
    step = torch.linalg.vector_norm(positions[:, :, 1:] - positions[:, :, :-1], dim=-1)
    assert step.max() <= 0.10001
    assert lengths.max() <= 20
