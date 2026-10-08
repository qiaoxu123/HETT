"""CPU regressions for terminal movement and episode observer accounting."""
import types
import unittest

import torch

from multiagent.trajectory_rollout_utils import should_record_pose
from scripts.joint_experiment_metrics import observe


class JointRolloutAccountingTest(unittest.TestCase):
    def test_terminal_movement_must_be_scored(self):
        start = (0., 0., 20., 0.)
        moved = (5., 0., 20., 0.)
        self.assertTrue(should_record_pose(start, moved, ended=True))
        self.assertFalse(should_record_pose(start, start, ended=True))
        self.assertTrue(should_record_pose(start, start, ended=False))

    def test_ended_episode_is_not_counted_again(self):
        # An already-ended episode must not contaminate per-step statistics
        # while other batch members continue rolling out.
        episode = {'experiment_steps': [{'ended': True, 'stopped': False}]}
        agent = types.SimpleNamespace(
            feedback='student', env_name='val_seen',
            args=types.SimpleNamespace(map_meters=410., success_dist=20.,
                heatmap_grid_size=7, trajectory_selector_mode='joint',
                trajectory_use_for_control=True),
        )
        proposals = types.SimpleNamespace(
            goal_xy=torch.zeros(1, 2, 2),
            mode_logits=torch.zeros(1, 2, 3),
            goal_ids=torch.zeros(1, 2, dtype=torch.long),
        )
        observe(agent, [None], [None], [episode], [True], 2, None,
                torch.zeros(1, 2, dtype=torch.long), proposals,
                torch.zeros(1, 6), torch.zeros(1, 6),
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long), None)
        self.assertEqual(len(episode['experiment_steps']), 1)


if __name__ == '__main__':
    unittest.main()
