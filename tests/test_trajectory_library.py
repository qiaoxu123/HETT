import inspect

import torch

from independent_nav.trajectory_library import (
    generate_rule_trajectory_set, stack_trajectory_set,
)


def test_rule_library_has_eight_budgeted_target_free_hypotheses():
    modes = [(0.8, 0.2), (0.8, 0.6), (0.5, 0.8),
             (0.2, 0.7), (0.3, 0.3)]
    values = generate_rule_trajectory_set((0.1, 0.1), 0.0, modes)
    stacked = stack_trajectory_set(values)
    assert len(values) == 8
    assert stacked["actions"].shape == (8, 20, 4)
    assert stacked["valid"].shape == (8, 20)
    assert stacked["positions"].shape == (8, 21, 2)
    assert stacked["movement_steps"].max() <= 20
    assert torch.all(stacked["actions"][..., :2].abs() <= 1.0 + 1e-6)
    assert "target" not in inspect.signature(
        generate_rule_trajectory_set).parameters


def test_direct_hypotheses_end_at_distinct_modes_and_views_are_distinct():
    modes = [(0.7, 0.2), (0.7, 0.6), (0.5, 0.8),
             (0.2, 0.7), (0.3, 0.3)]
    values = generate_rule_trajectory_set((0.1, 0.1), 0.0, modes)
    for index in range(5):
        assert torch.allclose(
            values[index].positions[-1], torch.tensor(modes[index]), atol=1e-6)
    assert not torch.allclose(values[5].positions, values[6].positions)
    assert not torch.allclose(values[6].positions, values[7].positions)

