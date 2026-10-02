import math
import unittest

import numpy as np

from multiagent.human_intent import (
    build_fixed_horizon_intent_targets,
    reconstruct_human_intent_path,
)
from multiagent.space import Pose4D


class HumanIntentTest(unittest.TestCase):
    def test_fixed_horizons_have_stable_physical_meaning(self):
        trajectory = [
            Pose4D(float(x), 0.0, 50.0, 0.0)
            for x in range(0, 41, 5)
        ]
        intent = reconstruct_human_intent_path(
            trajectory,
            min_step_m=0.0,
            rdp_tolerance_m=0.1,
            yaw_keyframe_deg=180.0,
        )
        targets = build_fixed_horizon_intent_targets(
            intent,
            current_xy=(5.0, 0.0),
            horizons_m=(10.0, 25.0, 50.0),
            goal_xy=(40.0, 0.0),
        )

        np.testing.assert_allclose(targets["xy"][0], [15.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(targets["xy"][1], [30.0, 0.0], atol=1e-5)
        self.assertEqual(targets["valid"].tolist(), [1.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(targets["xy"][-1], [40.0, 0.0], atol=1e-5)

    def test_large_view_rotation_is_preserved(self):
        trajectory = [
            Pose4D(0.0, 0.0, 50.0, 0.0),
            Pose4D(0.1, 0.0, 50.0, math.pi / 2),
            Pose4D(10.0, 0.0, 50.0, math.pi / 2),
        ]
        intent = reconstruct_human_intent_path(
            trajectory,
            min_step_m=1.0,
            rdp_tolerance_m=5.0,
            yaw_keyframe_deg=30.0,
        )

        self.assertEqual(len(intent), 3)
        self.assertAlmostEqual(intent[1].yaw, math.pi / 2, places=6)

    def test_small_keyboard_jitter_is_removed(self):
        trajectory = [
            Pose4D(0.0, 0.0, 50.0, 0.0),
            Pose4D(2.0, 0.2, 50.0, 0.01),
            Pose4D(4.0, -0.2, 50.0, -0.01),
            Pose4D(6.0, 0.1, 50.0, 0.0),
            Pose4D(10.0, 0.0, 50.0, 0.0),
        ]
        intent = reconstruct_human_intent_path(
            trajectory,
            min_step_m=0.0,
            rdp_tolerance_m=0.5,
            yaw_keyframe_deg=45.0,
        )

        self.assertLess(len(intent), len(trajectory))
        self.assertEqual(intent[0], trajectory[0])
        self.assertEqual(intent[-1], trajectory[-1])


if __name__ == "__main__":
    unittest.main()
