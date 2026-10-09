"""Minimal CPU regressions for Goal/Path soft ranking and fixed waypoints."""
import unittest
import torch
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
            current, goals, modes=3, waypoints=8, local_step_norm=20/410)
        self.assertEqual(tuple(paths.shape), (2, 2, 3, 8, 2))
        self.assertTrue(torch.isfinite(paths).all())
        self.assertTrue(torch.all((paths >= 0) & (paths <= 1)))
        self.assertGreater((paths[:, :, 0, -1] - paths[:, :, 2, -1]).abs().sum().item(), 0)
        self.assertLessEqual(
            ((paths[..., -1, :] - current[:, None, None, :]).norm(dim=-1) * 410).max().item(), 21.0)

    def test_compact_no_oracle_dependency_no_learned_stop(self):
        head = HeatmapTrajectoryHead(feature_dim=16, hidden_dim=32, modes=3, waypoints=8)
        features = torch.randn(2, 16, 9, 9)
        prob = torch.ones(2, 9, 9) / 81
        current = torch.tensor([[0.15, 0.20], [0.70, 0.80]])
        heading = torch.tensor([[0., 1.], [0., 1.]])
        kwargs = dict(top_k=4, compact=True, map_meters=410, local_step_m=20)
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
        probabilities = torch.ones(2, 9, 9, requires_grad=True) / 81
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
