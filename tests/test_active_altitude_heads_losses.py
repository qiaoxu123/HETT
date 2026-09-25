import torch

from multiagent.active_altitude.losses import active_altitude_losses
from multiagent.models.active_altitude_heads import ActiveAltitudeHeads


def test_heads_and_losses_form_a_trainable_end_to_end_path():
    torch.manual_seed(0)
    heads = ActiveAltitudeHeads(hidden_size=16, dropout=0.0)
    outputs = heads(torch.randn(3, 16))

    assert outputs["altitude_action_logits"].shape == (3, 3)
    assert outputs["phase_logits"].shape == (3, 4)
    assert outputs["stop_logit"].shape == (3, 1)
    assert outputs["goal_log_variance"].shape == (3, 1)

    losses = active_altitude_losses(
        outputs,
        altitude_action_target=torch.tensor([0, 1, 2]),
        phase_target=torch.tensor([0, 1, 3]),
        stop_target=torch.tensor([0.0, 0.0, 1.0]),
        goal_error_squared=torch.tensor([4.0, 1.0, 0.25]),
        active=torch.tensor([True, False, True]),
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert heads.shared[0].weight.grad is not None
    assert torch.isfinite(heads.shared[0].weight.grad).all()


def test_losses_are_zero_and_differentiable_when_every_episode_is_inactive():
    heads = ActiveAltitudeHeads(hidden_size=8, dropout=0.0)
    outputs = heads(torch.randn(2, 8))
    losses = active_altitude_losses(
        outputs,
        altitude_action_target=torch.tensor([0, 1]),
        phase_target=torch.tensor([0, 1]),
        stop_target=torch.tensor([0.0, 1.0]),
        goal_error_squared=torch.tensor([1.0, 1.0]),
        active=torch.tensor([False, False]),
    )
    assert losses.total.item() == 0.0
    losses.total.backward()
