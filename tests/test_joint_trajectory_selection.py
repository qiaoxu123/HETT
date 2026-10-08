"""Regression checks for goal-first selection and arrival labels."""
import unittest
import torch
from multiagent.models.heatmap_trajectory import (
    select_goal_mode_indices, arrival_stop_targets,
)


class JointSelectionRegression(unittest.TestCase):
    def test_goal_probability_is_not_replaced_by_peaked_mode(self):
        # Goal 0 is more likely in aggregate; goal 1 has a larger single
        # mode probability. The previous flattened argmax picked goal 1.
        scores = torch.log(torch.tensor([[
            [0.40, 0.30, 0.10], [0.19, 0.005, 0.005]
        ]]))
        self.assertEqual(int(select_goal_mode_indices(scores)[0]), 0)
        self.assertEqual(int(scores.flatten(1).argmax(-1)[0]), 0)
        # More adversarial case: goal 0 has mass 0.6 spread across modes.
        scores = torch.log(torch.tensor([[
            [.21, .20, .19], [.39, .005, .005]
        ]]))
        self.assertEqual(int(select_goal_mode_indices(scores)[0]), 0)
        self.assertEqual(int(scores.flatten(1).argmax(-1)[0]), 3)

    def test_goal_mode_shape_and_multi_batch(self):
        scores = torch.tensor([[[0., 0.], [1., 2.]],
                               [[3., 0.], [1., 1.]]])
        result = select_goal_mode_indices(scores)
        self.assertEqual(result.tolist(), [3, 0])
        with self.assertRaises(ValueError):
            select_goal_mode_indices(torch.zeros(2, 3))

    def test_stop_target_matches_official_success_radius(self):
        current = torch.tensor([[0., 0.], [0., 0.], [.1, 0.]])
        goal = torch.tensor([[.02, 0.], [.06, 0.], [.1, 0.]])
        labels = arrival_stop_targets(
            current, goal, map_meters=410., success_radius_m=20.)
        self.assertEqual(labels.tolist(), [1., 0., 1.])
        with self.assertRaises(ValueError):
            arrival_stop_targets(current, goal, map_meters=0, success_radius_m=20.)


if __name__ == '__main__':
    unittest.main()
