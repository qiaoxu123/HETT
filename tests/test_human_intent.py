import math
import unittest

import numpy as np

from multiagent.human_intent import (
    build_next_core_anchor_target,
    extract_human_core_anchors,
)
from multiagent.space import Pose4D


class HumanCoreAnchorTest(unittest.TestCase):
    def test_spatial_turn_is_preserved_as_core_anchor(self):
        trajectory = [
            Pose4D(0.0, 0.0, 50.0, 0.0),
            Pose4D(10.0, 0.0, 50.0, 0.0),
            Pose4D(20.0, 0.0, 50.0, 0.0),
            Pose4D(20.0, 10.0, 50.0, math.pi / 2),
            Pose4D(20.0, 20.0, 50.0, math.pi / 2),
        ]
        core = extract_human_core_anchors(
            trajectory,
            rdp_tolerance_m=1.0,
            yaw_keyframe_deg=180.0,
        )
        self.assertIn(2, core.indices)
        pos = core.indices.index(2)
        self.assertIn("turn", core.reasons[pos])

    def test_large_view_change_is_preserved(self):
        trajectory = [
            Pose4D(0.0, 0.0, 50.0, 0.0),
            Pose4D(0.2, 0.0, 50.0, math.pi / 2),
            Pose4D(10.0, 0.0, 50.0, math.pi / 2),
        ]
        core = extract_human_core_anchors(
            trajectory,
            min_step_m=1.0,
            rdp_tolerance_m=10.0,
            yaw_keyframe_deg=30.0,
        )
        self.assertIn(1, core.indices)
        pos = core.indices.index(1)
        self.assertIn("yaw", core.reasons[pos])

    def test_referenced_landmark_passage_becomes_anchor(self):
        trajectory = [
            Pose4D(float(x), 0.0, 50.0, 0.0)
            for x in range(0, 51, 10)
        ]
        core = extract_human_core_anchors(
            trajectory,
            landmark_centroids_xy=[(31.0, 4.0)],
            rdp_tolerance_m=100.0,
            yaw_keyframe_deg=180.0,
            landmark_radius_m=10.0,
        )
        self.assertIn(3, core.indices)
        pos = core.indices.index(3)
        self.assertIn("landmark", core.reasons[pos])

    def test_next_anchor_limits_trajectory_supervision(self):
        trajectory = [
            Pose4D(0.0, 0.0, 50.0, 0.0),
            Pose4D(10.0, 0.0, 50.0, 0.0),
            Pose4D(20.0, 0.0, 50.0, 0.0),
            Pose4D(20.0, 10.0, 50.0, math.pi / 2),
            Pose4D(20.0, 20.0, 50.0, math.pi / 2),
        ]
        core = extract_human_core_anchors(
            trajectory,
            rdp_tolerance_m=1.0,
            yaw_keyframe_deg=180.0,
        )
        target = build_next_core_anchor_target(
            trajectory,
            core,
            current_xy=(0.0, 0.0),
            horizons_m=(10.0, 25.0, 50.0, 100.0),
            min_lookahead_m=8.0,
            current_arc_m=0.0,
        )
        np.testing.assert_allclose(
            target["anchor_xy"],
            [20.0, 0.0],
            atol=1e-5,
        )
        self.assertEqual(
            target["valid"].tolist(),
            [1.0, 0.0, 0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            target["trajectory_xy"][-1],
            target["anchor_xy"],
            atol=1e-5,
        )

    def test_known_teacher_arc_avoids_self_intersection_projection(self):
        trajectory = [
            Pose4D(0.0, 0.0, 50.0, 0.0),
            Pose4D(10.0, 0.0, 50.0, 0.0),
            Pose4D(0.0, 0.0, 50.0, math.pi),
            Pose4D(-10.0, 0.0, 50.0, math.pi),
        ]
        core = extract_human_core_anchors(
            trajectory,
            rdp_tolerance_m=0.1,
            yaw_keyframe_deg=30.0,
        )
        target = build_next_core_anchor_target(
            trajectory,
            core,
            current_xy=(0.0, 0.0),
            horizons_m=(5.0,),
            min_lookahead_m=4.0,
            current_arc_m=20.0,
        )
        self.assertGreaterEqual(float(target["anchor_arc_m"]), 20.0)


if __name__ == "__main__":
    unittest.main()
