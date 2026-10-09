"""Minimal CPU regressions for Goal/Path soft ranking and fixed waypoints."""
import unittest
import torch
from multiagent.heatmap_execution import bounded_heatmap_step
from multiagent.space import Pose4D
from multiagent.models.heatmap_trajectory import (
    HeatmapTrajectoryHead,
    fixed_local_anchor_paths,
    goal_distance_soft_ranking_loss,
    local_path_soft_ranking_loss,
)


class CompactGoalPathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_fixed_anchors_have_distinct_bounded_endpoints(self):
        current = torch.tensor([[0.25, 0.30], [0.80, 0.75]])
        goals = torch.tensor([[[0.75, 0.30], [0.20, 0.60]],
                              [[0.50, 0.80], [0.95, 0.35]]])
        paths = fixed_local_anchor_paths(
            current, goals, modes=3, waypoints=8, local_step_norm=50/410)
        self.assertEqual(tuple(paths.shape), (2, 2, 3, 8, 2))
        self.assertTrue(torch.isfinite(paths).all())
        self.assertTrue(torch.all((paths >= 0) & (paths <= 1)))
        self.assertGreater((paths[:, :, 0, -1] - paths[:, :, 2, -1]).abs().sum().item(), 0)
        step_lengths_m = ((paths[..., -1, :] -
                           current[:, None, None, :]).norm(dim=-1) * 410)
        self.assertLessEqual(step_lengths_m.max().item(), 50.001)
        # Every mode must have the same step length when the goal is far
        # and the endpoint is away from map boundaries.
        self.assertTrue(torch.allclose(
            step_lengths_m[0, 0], torch.full((3,), 50.), atol=1e-3))

    def test_close_goal_has_no_lateral_detour(self):
        current = torch.tensor([[.40, .40]])
        goals = torch.tensor([[[.42, .42], [.45, .40]]])
        paths = fixed_local_anchor_paths(
            current, goals, modes=3, waypoints=8, local_step_norm=50/410)
        # At distances below 50m all paths directly reach the same goal.
        self.assertTrue(torch.allclose(
            paths[..., -1, :], goals[:, :, None, :].expand(-1, -1, 3, -1), atol=1e-6))

    def test_long_distance_reachable_with_same_twenty_step_budget(self):
        # ~402m initial distance: previously impossible under 20x20m cap.
        goal = torch.tensor([[[.99, .50]]])
        origin = torch.tensor([[.01, .50]])
        pos = origin.clone()
        steps = []
        for _ in range(20):
            anchors = fixed_local_anchor_paths(
                pos, goal, modes=3, waypoints=8, local_step_norm=50/410)
            # Even repeatedly choosing the left anchor does not impose a
            # shorter travel budget: each displacement may be up to 50m.
            target = anchors[0, 0, 0, -1] * 410
            pose = Pose4D(float(pos[0, 0] * 410), float(pos[0, 1] * 410), 50., 0.)
            next_pose = bounded_heatmap_step(pose, target.tolist(), max_step_m=50.)
            moved_m = pose.xy.dist_to(next_pose.xy)
            steps.append(moved_m)
            self.assertLessEqual(moved_m, 50.001)
            pos = torch.tensor([[next_pose.x / 410, next_pose.y / 410]])
            if torch.linalg.vector_norm(pos - goal[:, 0]).item() * 410 <= 20:
                break
        self.assertGreater(float(torch.linalg.vector_norm(origin - goal[:, 0]) * 410), 380.)
        self.assertLessEqual(len(steps), 20)
        self.assertLessEqual(float(torch.linalg.vector_norm(pos - goal[:, 0]) * 410), 20.)
        self.assertGreater(max(steps), 49.0)

    def test_local_path_scores_depend_on_current_state(self):
        head = HeatmapTrajectoryHead(feature_dim=16, hidden_dim=32, modes=3, waypoints=8)
        feature = torch.randn(2, 16, 9, 9)
        prob = torch.ones(2, 9, 9) / 81
        here = torch.tensor([[.20, .20], [.70, .60]])
        heading = torch.tensor([[0., 1.], [1., 0.]])
        pred, _ = head(feature, prob, here, heading,
                       compact=True, top_k=5, local_step_m=50)
        diff = pred.mode_logits[:, :, 0] - pred.mode_logits[:, :, 1]
        # Old additive mode embeddings with a linear score made the
        # difference invariant to candidate features and pose.
        self.assertGreater(float((diff[0] - diff[1]).abs().max()), 1e-6)

    def test_compact_no_oracle_dependency_no_learned_stop(self):
        head = HeatmapTrajectoryHead(feature_dim=16, hidden_dim=32, modes=3, waypoints=8)
        features = torch.randn(2, 16, 9, 9)
        prob = torch.ones(2, 9, 9) / 81
        current = torch.tensor([[0.15, 0.20], [0.70, 0.80]])
        heading = torch.tensor([[0., 1.], [0., 1.]])
        kwargs = dict(top_k=4, compact=True, map_meters=410, local_step_m=50)
        output, teacher = head(features, prob, current, heading, **kwargs)
        output_with_gt, teacher_with_gt = head(
            features, prob, current, heading, teacher_goal=torch.ones_like(current), **kwargs)
        self.assertIsNone(teacher)
        self.assertIsNone(teacher_with_gt)
        self.assertEqual(tuple(output.trajectories.shape), (2, 4, 3, 8, 2))
        self.assertTrue(torch.equal(output.trajectories, output_with_gt.trajectories))
        self.assertTrue(torch.equal(output.stop_logits, torch.zeros(2)))

    def test_soft_goal_loss_trains_all_far_samples_and_selector_only(self):
        head = HeatmapTrajectoryHead(feature_dim=16, hidden_dim=32, modes=3, waypoints=8)
        feature = torch.randn(2, 16, 9, 9, requires_grad=True)
        probabilities = torch.full((2, 9, 9), 1.0/81.0, requires_grad=True)
        here = torch.tensor([[.3, .2], [.6, .8]])
        output, _ = head(feature.detach(), probabilities.detach(), here,
                         torch.tensor([[0., 1.], [0., 1.]]), compact=True, top_k=5)
        target = torch.tensor([[0.99, 0.99], [0.01, 0.99]])
        loss = goal_distance_soft_ranking_loss(
            output, target, map_meters=410, temperature_m=20)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertIsNone(feature.grad)
        self.assertIsNone(probabilities.grad)
        self.assertIsNotNone(head.goal_scorer[-1].weight.grad)
        self.assertTrue(torch.isfinite(head.goal_scorer[-1].weight.grad).all())

    def test_path_soft_loss_trains_modes_and_masks_no_near_goal(self):
        head = HeatmapTrajectoryHead(feature_dim=16, hidden_dim=32, modes=3, waypoints=8)
        feat = torch.randn(2, 16, 9, 9)
        prob = torch.ones(2, 9, 9) / 81
        here = torch.tensor([[.20, .20], [.25, .25]])
        output, _ = head(feat, prob, here, torch.tensor([[0., 1.], [0., 1.]]),
                         compact=True, top_k=5)
        nearest = output.goal_xy[:, 0].detach().clone()
        teacher_next = here + torch.tensor([[.02, .015], [0., .02]])
        loss, count = local_path_soft_ranking_loss(
            output, nearest, here, teacher_next, map_meters=410)
        self.assertEqual(int(count), 2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(head.mode_score.weight.grad)
        self.assertGreater(head.mode_score.weight.grad.abs().sum().item(), 0)

        far = torch.tensor([[.99, .99], [.99, .99]])
        absent, count = local_path_soft_ranking_loss(
            output, far, here, teacher_next, map_meters=410,
            positive_radius_m=0.01)
        self.assertEqual(int(count), 0)
        self.assertEqual(float(absent), 0.0)

    def test_invalid_soft_temperature(self):
        head = HeatmapTrajectoryHead(feature_dim=16, hidden_dim=32)
        features = torch.randn(1, 16, 9, 9)
        p, _ = head(features, torch.ones(1, 9, 9)/81,
                    torch.tensor([[.3, .3]]), torch.tensor([[0., 1.]]),
                    compact=True, top_k=3)
        with self.assertRaises(ValueError):
            goal_distance_soft_ranking_loss(
                p, torch.tensor([[.3, .3]]), map_meters=410, temperature_m=0)
        with self.assertRaises(ValueError):
            local_path_soft_ranking_loss(
                p, torch.tensor([[.3, .3]]), torch.tensor([[.3, .3]]),
                torch.tensor([[.5, .3]]), map_meters=410, local_step_m=0)


if __name__ == "__main__":
    unittest.main()
