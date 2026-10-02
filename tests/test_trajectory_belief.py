import unittest

import numpy as np
import torch

from multiagent.trajectory_belief import (
    build_fixed_horizon_anchors,
    prepare_teacher_path,
    prepare_teacher_rollout_path,
    refine_candidate_endpoints,
    sample_fixed_horizon_targets,
    sample_fixed_horizon_targets_from_arc,
    sample_teacher_pose_at_arc,
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

    def test_endpoint_offset_refines_inside_7x7_cell(self):
        offsets = torch.zeros(1, 49, 2)
        center = refine_candidate_endpoints(offsets, 7)[0, 0]
        torch.testing.assert_close(center, torch.tensor([0.5 / 7, 0.5 / 7]))

        offsets[0, 0] = torch.tensor([0.5 / 7, -0.5 / 7])
        refined = refine_candidate_endpoints(offsets, 7)[0, 0]
        torch.testing.assert_close(refined, torch.tensor([1.0 / 7, 0.0]))

    def test_teacher_rollout_advances_by_metric_arc_length(self):
        path = [
            Pose4D(0.0, 0.0, 40.0, 0.0),
            Pose4D(6.0, 0.0, 50.0, 0.0),
            Pose4D(6.0, 8.0, 60.0, np.pi / 2),
        ]
        poses, cumulative = prepare_teacher_rollout_path(path)
        self.assertAlmostEqual(float(cumulative[-1]), 14.0, places=5)

        pose = sample_teacher_pose_at_arc(poses, cumulative, 10.0)
        self.assertAlmostEqual(pose.x, 6.0, places=5)
        self.assertAlmostEqual(pose.y, 4.0, places=5)
        self.assertAlmostEqual(pose.z, 55.0, places=5)
        self.assertAlmostEqual(pose.yaw, np.pi / 4, places=5)

    def test_teacher_yaw_interpolation_uses_short_wrapped_turn(self):
        path = [
            Pose4D(0.0, 0.0, 50.0, np.deg2rad(170.0)),
            Pose4D(10.0, 0.0, 50.0, np.deg2rad(-170.0)),
        ]
        poses, cumulative = prepare_teacher_rollout_path(path)
        pose = sample_teacher_pose_at_arc(poses, cumulative, 5.0)
        self.assertAlmostEqual(abs(pose.yaw), np.pi, places=5)

    def test_known_teacher_arc_avoids_projection_ambiguity(self):
        points = np.asarray(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 0.0], [-10.0, 0.0]],
            dtype=np.float32,
        )
        segments = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segments))).astype(np.float32)
        targets, _ = sample_fixed_horizon_targets_from_arc(
            points, cumulative, 20.0, (5.0,)
        )
        np.testing.assert_allclose(targets[0], [-5.0, 0.0], atol=1e-5)

    def test_nms_operates_directly_on_7x7_modes(self):
        probs = torch.zeros(1, 49)
        probs[0, 1 * 7 + 1] = 0.9
        probs[0, 1 * 7 + 2] = 0.8
        probs[0, 5 * 7 + 5] = 0.7
        ids, scores, _ = select_nms_topk(probs, 7, 2, 3)
        self.assertEqual(int(ids[0, 0]), 1 * 7 + 1)
        self.assertEqual(int(ids[0, 1]), 5 * 7 + 5)
        self.assertGreater(float(scores[0, 0]), float(scores[0, 1]))


if __name__ == "__main__":
    unittest.main()
