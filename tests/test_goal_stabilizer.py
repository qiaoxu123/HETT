import unittest
import numpy as np
from multiagent.goal_stabilizer import stabilize_goals


class GoalStabilizerTests(unittest.TestCase):
    def setUp(self):
        self.prev_xy = np.array([[.1, .1], [.2, .2], [.3, .3]], dtype=np.float32)
        self.new_xy = np.array([[.5, .5], [.6, .6], [.7, .7]], dtype=np.float32)
        self.scores = np.log(np.full((3, 4), .25))
        self.scores[:, 2] = np.log(.5)

    def test_disabled_is_identity(self):
        xy, cell = stabilize_goals(self.prev_xy, np.array([0, 1, -1]), self.new_xy,
                                   np.array([2, 2, 2]), self.scores, np.array([5., 50., 5.]))
        np.testing.assert_array_equal(xy, self.new_xy)
        np.testing.assert_array_equal(cell, [2, 2, 2])

    def test_lock_keeps_goal_near_arrival_only(self):
        xy, cell = stabilize_goals(self.prev_xy, np.array([0, 1, -1]), self.new_xy,
                                   np.array([2, 2, 2]), self.scores, np.array([5., 50., 5.]),
                                   lock_m=20)
        np.testing.assert_array_equal(cell, [0, 2, 2])
        np.testing.assert_array_equal(xy[0], self.prev_xy[0])
        np.testing.assert_array_equal(xy[1], self.new_xy[1])

    def test_margin_requires_clear_improvement(self):
        # log(.5) - log(.25) = 0.69 nats
        xy, cell = stabilize_goals(self.prev_xy, np.array([0, 1, 1]), self.new_xy,
                                   np.array([2, 2, 2]), self.scores, np.array([99., 99., 99.]),
                                   margin=1.0)
        np.testing.assert_array_equal(cell, [0, 1, 1])
        _, cell = stabilize_goals(self.prev_xy, np.array([0, 1, 1]), self.new_xy,
                                  np.array([2, 2, 2]), self.scores, np.array([99., 99., 99.]),
                                  margin=0.5)
        np.testing.assert_array_equal(cell, [2, 2, 2])

    def test_prev_not_in_candidate_pool_switches(self):
        scores = self.scores.copy(); scores[:, 0] = -np.inf
        _, cell = stabilize_goals(self.prev_xy, np.array([0, 0, 0]), self.new_xy,
                                  np.array([2, 2, 2]), scores, np.array([99., 99., 99.]),
                                  margin=5.0)
        np.testing.assert_array_equal(cell, [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
