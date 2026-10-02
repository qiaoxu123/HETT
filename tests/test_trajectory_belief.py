import unittest

import numpy as np
import torch

from multiagent.trajectory_belief import (
    build_fixed_horizon_anchors,
    prepare_teacher_path,
    sample_fixed_horizon_targets,
    select_nms_topk,
)
from multiagent.space import Pose4D


class TrajectoryBeliefUtilityTest(unittest.TestCase):
    def test_fixed_horizon_targets_have_metric_semantics(self):
        path = [Pose4D(float(x), 0.0, 50.0, 0.0) for x in range(0, 121, 10)]
        points, cumulative = prepare_teacher_path(path, (120.0, 0.0))
        targets, valid = sample_fixed_horizon_targets(
            points, cumulative, (5.0, 0.0), (10.0, 25.0, 50.0, 100.0)
        )
        np.testing.assert_allclose(
            targets[:, 0], [15.0, 30.0, 55.0, 105.0, 120.0], atol=1e-5
        )
        self.assertEqual(valid.tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_short_remaining_path_masks_long_horizons(self):
        path = [Pose4D(0.0, 0.0, 50.0, 0.0), Pose4D(20.0, 0.0, 50.0, 0.0)]
        points, cumulative = prepare_teacher_path(path, (20.0, 0.0))
        targets, valid = sample_fixed_horizon_targets(
            points, cumulative, (5.0, 0.0), (10.0, 25.0, 50.0, 100.0)
        )
        self.assertEqual(valid.tolist(), [1.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(targets[-1], [20.0, 0.0], atol=1e-5)

    def test_anchor_first_waypoint_is_ten_meters_on_410m_map(self):
        current = torch.tensor([[0.1, 0.1]])
        endpoints = torch.tensor([[[0.9, 0.1]]])
        anchors = build_fixed_horizon_anchors(
            current, endpoints, (10.0, 25.0, 50.0, 100.0), 410.0
        )
        first_distance = torch.linalg.norm(anchors[0, 0, 0] - current[0]) * 410.0
        self.assertAlmostEqual(float(first_distance), 10.0, places=4)
        torch.testing.assert_close(anchors[0, 0, -1], endpoints[0, 0])

    def test_nms_returns_spatially_separated_peaks(self):
        probs = torch.zeros(1, 28 * 28)
        probs[0, 5 * 28 + 5] = 0.9
        probs[0, 5 * 28 + 6] = 0.8
        probs[0, 20 * 28 + 20] = 0.7
        ids, scores, _ = select_nms_topk(probs, 28, 2, 5)
        self.assertEqual(int(ids[0, 0]), 5 * 28 + 5)
        self.assertEqual(int(ids[0, 1]), 20 * 28 + 20)
        self.assertGreater(float(scores[0, 0]), float(scores[0, 1]))


if __name__ == "__main__":
    unittest.main()
