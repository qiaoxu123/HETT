import torch

from independent_nav.trajectory_utility import geometric_trajectory_utility


def test_geometric_utility_prefers_jointly_visible_path_and_is_target_free():
    positions = torch.tensor([[[
        [0.0, 0.0], [0.2, 0.2], [0.4, 0.4]],
        [[0.0, 0.0], [0.0, 0.1], [0.0, 0.2]],
    ]])
    result = geometric_trajectory_utility(
        positions, torch.ones(1, 2, 3, dtype=torch.bool),
        torch.tensor([[[0.45, 0.45], [0.45, 0.45]]]),
        torch.tensor([[[0.5, 0.5]]]), torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[[0.45, 0.45], [0.55, 0.45]]]))
    assert result["joint_visibility"][0, 0] > result["joint_visibility"][0, 1]
    assert result["utility"].shape == (1, 2)
