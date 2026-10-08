import unittest
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F

from multiagent.models.heatmap_trajectory import (
    HeatmapTrajectoryHead, heatmap_endpoints, resample_teacher_suffix,
    trajectory_imitation_loss, stop_supervision_loss,
    candidate_ranking_loss, candidate_trajectory_imitation_loss,
)


@dataclass
class P:
    x: float
    y: float


@dataclass
class Pose:
    xy: P


@dataclass
class Bounds:
    x_min: float
    y_max: float


class HeatmapTrajectoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_topk_goal_ids_are_column_x_row_y(self):
        belief = torch.zeros(1, 7, 7)
        belief[0, 2, 5] = 0.7
        belief[0, 5, 1] = 0.3
        xy, ids = heatmap_endpoints(belief, 2)
        self.assertEqual(ids.shape, (1, 2))
        self.assertEqual(int(ids[0, 0]), 19)
        self.assertAlmostEqual(float(xy[0, 0, 0]), 5.5/7)
        self.assertAlmostEqual(float(xy[0, 0, 1]), 2.5/7)

    def test_modes_preserve_endpoints_and_are_distinct(self):
        head = HeatmapTrajectoryHead(feature_dim=32, hidden_dim=48,
                                     modes=3, waypoints=8).eval()
        features = torch.randn(2, 32, 14, 14)
        belief = torch.ones(2, 14, 14)/196
        pose = torch.tensor([[0.2,0.2], [0.8,0.8]])
        heading = torch.tensor([[0.,1.], [1.,0.]])
        output, teacher = head(features, belief, pose, heading, top_k=5)
        self.assertIsNone(teacher)
        self.assertEqual(output.trajectories.shape, (2,5,3,8,2))
        self.assertEqual(output.mode_logits.shape, (2,5,3))
        self.assertEqual(output.joint_logits.shape, (2,5,3))
        self.assertTrue(torch.allclose(output.trajectories[..., -1, :],
                                       output.goal_xy[:, :, None].expand(-1,-1,3,-1)))
        self.assertTrue(torch.isfinite(output.trajectories).all())
        self.assertGreater(float((
            output.trajectories[:, :, 0, 3] -
            output.trajectories[:, :, -1, 3]).abs().max()), 1e-4)

    def test_teacher_conditioning_is_optional_and_gradients_flow(self):
        head = HeatmapTrajectoryHead(feature_dim=32, hidden_dim=48,
                                     modes=3, waypoints=8)
        features = torch.randn(2, 32, 14, 14, requires_grad=True)
        belief = torch.softmax(torch.randn(2, 14*14),-1).reshape(2,14,14)
        here = torch.tensor([[.1,.1],[.8,.7]])
        target = torch.tensor([[.7,.7],[.2,.1]])
        model, teacher = head(features, belief, here,
                              torch.tensor([[0.,1.],[0.,1.]]),
                              top_k=5, teacher_goal=target)
        self.assertEqual(teacher[0].shape, (2,3,8,2))
        self.assertTrue(torch.allclose(
            teacher[0][:, :, -1], target[:, None].expand(-1,3,-1)))
        gt_path = here[:,None,:] + torch.linspace(.125,1.,8)[None,:,None] * (
            target - here)[:, None, :]
        l = trajectory_imitation_loss(teacher, gt_path)
        l = l + .1*stop_supervision_loss(model.stop_logits,
                                         torch.tensor([0., 1.]))
        l.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertIsNotNone(head.residual[-1].weight.grad)
        self.assertTrue(torch.isfinite(head.residual[-1].weight.grad).all())

    def test_suffix_resampling_extends_human_endpoint_to_true_goal(self):
        traj = [Pose(P(0., 0.)), Pose(P(5., 0.)), Pose(P(10., 0.))]
        bounds = {"city": Bounds(x_min=0., y_max=100.)}
        target = resample_teacher_suffix(
            traj, [0., 1.], map_name="city", bounds=bounds,
            map_meters=100., steps=8, goal_xy=[.2, 1.],
        )
        self.assertEqual(tuple(target.shape), (8,2))
        self.assertTrue(torch.allclose(target[-1],torch.tensor([.2,1.])))
        self.assertGreater(float(target[-1,0]), float(target[0,0]))

    def test_invalid_lengths_and_loss_shape(self):
        with self.assertRaises(ValueError):
            HeatmapTrajectoryHead(waypoints=1)
        with self.assertRaises(ValueError):
            resample_teacher_suffix([], [0.,0.], map_name="x",
                                    bounds={}, map_meters=100, steps=8)
        with self.assertRaises(ValueError):
            trajectory_imitation_loss(
                (torch.zeros(1,3,8,2), torch.zeros(1,3)), torch.zeros(1,7,2)
            )


    def test_joint_scorer_starts_identical_to_prior_goal_ranking(self):
        head = HeatmapTrajectoryHead(feature_dim=16, hidden_dim=32,
                                     modes=3, waypoints=6)
        feature = torch.randn(2, 16, 9, 9)
        prob = torch.softmax(torch.randn(2, 81), dim=-1).reshape(2, 9, 9)
        here = torch.tensor([[.2, .2], [.7, .2]])
        heading = torch.tensor([[0., 1.], [1., 0.]])
        pred, teacher = head(feature, prob, here, heading, top_k=4)
        self.assertIsNone(teacher)
        self.assertEqual(tuple(pred.candidate_logits.shape), (2, 4))
        log_prior = prob.flatten(1).gather(1, pred.goal_ids).log()
        baseline = (F.log_softmax(log_prior, dim=-1)[:, :, None]
                    + F.log_softmax(pred.mode_logits, dim=-1))
        self.assertTrue(torch.allclose(pred.joint_logits, baseline, atol=1e-6))

    def test_predicted_goal_ranking_loss_updates_scorer(self):
        head = HeatmapTrajectoryHead(feature_dim=16, hidden_dim=32,
                                     modes=3, waypoints=6)
        feature = torch.randn(2, 16, 9, 9, requires_grad=True)
        prob = torch.ones(2, 9, 9) / 81
        here = torch.tensor([[.2, .2], [.2, .2]])
        heading = torch.tensor([[0., 1.], [0., 1.]])
        pred, _ = head(feature, prob, here, heading, top_k=4)
        target = pred.goal_xy[:, 1].detach().clone()
        loss = candidate_ranking_loss(
            pred, target, map_meters=410., positive_radius_m=20.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(head.goal_scorer[-1].weight.grad)
        self.assertGreater(float(head.goal_scorer[-1].weight.grad.abs().sum()), 0.)

    def test_candidate_path_supervision_and_invalid_candidate_mask(self):
        head = HeatmapTrajectoryHead(feature_dim=16, hidden_dim=32,
                                     modes=3, waypoints=6)
        feature = torch.randn(1, 16, 9, 9)
        prob = torch.zeros(1, 9, 9)
        prob[0, 3, 4] = .8
        prob[0, 7, 7] = .2
        here = torch.tensor([[.2, .2]])
        heading = torch.tensor([[0., 1.]])
        pred, _ = head(feature, prob, here, heading, top_k=2)
        target = pred.goal_xy[:, 0].detach()
        steps = torch.linspace(1 / 6., 1., 6)[None, :, None]
        teacher = here[:, None] + steps * (target[:, None] - here[:, None])
        path_loss = candidate_trajectory_imitation_loss(
            pred, target, teacher, map_meters=410., positive_radius_m=20.)
        self.assertTrue(torch.isfinite(path_loss))
        path_loss.backward()
        self.assertIsNotNone(head.residual[-1].weight.grad)
        self.assertGreater(float(head.residual[-1].weight.grad.abs().sum()), 0.)
        far = torch.tensor([[0., 0.]])
        no_positive = candidate_ranking_loss(
            pred, far, map_meters=410., positive_radius_m=1.)
        self.assertAlmostEqual(float(no_positive), 0.)
        no_path_positive = candidate_trajectory_imitation_loss(
            pred, far, teacher, map_meters=410., positive_radius_m=1.)
        self.assertAlmostEqual(float(no_path_positive), 0.)

if __name__ == "__main__":
    unittest.main()
