import math
import unittest

from multiagent.heatmap_execution import bounded_heatmap_step
from multiagent.space import Pose4D


class BoundedHeatmapExecutionTest(unittest.TestCase):
    def test_caps_horizontal_step_and_preserves_altitude(self):
        pose = Pose4D(10.0, 20.0, 35.0, 0.2)
        moved = bounded_heatmap_step(pose, [110.0, 20.0], max_step_m=30.0)
        self.assertEqual(moved, Pose4D(40.0, 20.0, 35.0, 0.0))

    def test_reaches_nearby_waypoint(self):
        pose = Pose4D(0.0, 0.0, 10.0, 1.0)
        moved = bounded_heatmap_step(pose, [3.0, 4.0], max_step_m=10.0)
        self.assertAlmostEqual(moved.x, 3.0)
        self.assertAlmostEqual(moved.y, 4.0)
        self.assertAlmostEqual(moved.yaw, math.atan2(4.0, 3.0))

    def test_rejects_invalid_step_length(self):
        with self.assertRaises(ValueError):
            bounded_heatmap_step(Pose4D(0.0, 0.0, 0.0, 0.0), [1.0, 0.0], 0)


if __name__ == "__main__":
    unittest.main()
