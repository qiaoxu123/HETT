"""CPU equivalence/instrumentation checks for the layered HETT fastpath."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from multiagent.space import Point2D
from multiagent.static_observation import build_static_landmark_observation
from multiagent.models import heatmap_trajectory as traj_module
from multiagent.models.heatmap_trajectory import HeatmapTrajectoryHead
from multiagent.models.candidate_relation_selector import CandidateRelationSelector


class StaticObservationTests(unittest.TestCase):
    def test_name_geometry_pairing_matches_original_normalization(self):
        triangle = [Point2D(10., 80.), Point2D(20., 80.), Point2D(10., 70.)]
        matched = SimpleNamespace(contour=triangle)
        referenced = SimpleNamespace(landmarks=[matched], landmark_names=['building'])
        nav_map = SimpleNamespace(
            landmark_map=SimpleNamespace(get_contours=lambda: [triangle]),
            referenced_landmark_map=referenced,
        )
        calls = []
        def normalize(position, map_name, meters):
            calls.append((map_name, meters))
            return ((position.x - 0.) / meters, (100. - position.y) / meters)
        output = build_static_landmark_observation(nav_map, 'city', 100., normalize)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(output['reference_landmarks']), 1)
        self.assertEqual(output['reference_landmarks'][0]['name'], 'building')
        np.testing.assert_allclose(
            output['reference_landmarks'][0]['center_xy'],
            [(.1+.2+.1)/3., (.2+.2+.3)/3.], atol=1e-6)
        np.testing.assert_allclose(output['reference_landmarks'][0]['extent_xy'],
                                   [.1,.1], atol=1e-6)
        np.testing.assert_allclose(output['centroids'],
                                   [(.1+.2+.1)/3., (.2+.2+.3)/3.], atol=1e-6)

    def test_no_matched_anchors_retains_legacy_empty_fallback(self):
        nav_map = SimpleNamespace(
            landmark_map=SimpleNamespace(get_contours=lambda: []),
            referenced_landmark_map=SimpleNamespace(landmarks=[], landmark_names=[]))
        output = build_static_landmark_observation(
            nav_map, 'city', 410., lambda *_: (_ for _ in ()).throw(AssertionError()))
        self.assertEqual(output['reference_landmarks'], [])
        self.assertTrue(np.array_equal(output['centroids'], np.array([0,0])))
        self.assertTrue(np.array_equal(output['centroid_goal'], np.array([0,0])))


class PlanningEquivalenceTests(unittest.TestCase):
    def test_candidate_grid_features_sampled_once_not_twice(self):
        torch.manual_seed(3)
        head = HeatmapTrajectoryHead(
            feature_dim=16, language_dim=12, hidden_dim=32, modes=3, waypoints=6).eval()
        feature = torch.randn(1,16,9,9,requires_grad=True)
        prob = torch.softmax(torch.randn(1,81),-1).reshape(1,9,9)
        pose = torch.tensor([[.2,.3]])
        yaw = torch.tensor([[0.,1.]])
        language = torch.randn(1,5,12)
        landmark = torch.tensor([[[.5,.5]]])
        kwargs = dict(top_k=3, language_tokens=language,
                      language_mask=torch.ones(1,5,dtype=torch.bool),
                      landmark_xy=landmark, landmark_extent=torch.ones_like(landmark)*.2,
                      landmark_valid=torch.ones(1,1,dtype=torch.bool),
                      landmark_text_mask=torch.ones(1,1,5,dtype=torch.bool))
        gather = traj_module._gather_heatmap_features
        with patch.object(traj_module, '_gather_heatmap_features', wraps=gather) as fn:
            proposal, teacher = head(feature,prob,pose,yaw,**kwargs)
        self.assertIsNone(teacher)
        # Once for proposals, once for the Stop head at the current UAV pose.
        self.assertEqual(fn.call_count, 2)
        loss = proposal.joint_logits.sum() + proposal.trajectories.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(feature.grad).all())

    def test_language_key_value_projection_is_shared(self):
        torch.manual_seed(5)
        selector=CandidateRelationSelector(
            feature_dim=16,language_dim=12,hidden_dim=24,attention_heads=4,dropout=0.).eval()
        calls=[]
        hook=selector.token_proj.register_forward_hook(
            lambda module, inputs, output: calls.append(output.shape))
        try:
            selector(goal_xy=torch.tensor([[[.3,.3],[.7,.4]]]),
                     goal_features=torch.randn(1,2,16),
                     current_xy=torch.tensor([[.1,.1]]),
                     heading_sc=torch.tensor([[0.,1.]]),
                     language_tokens=torch.randn(1,4,12),
                     language_mask=torch.ones(1,4,dtype=torch.bool),
                     landmark_xy=torch.tensor([[[.5,.5]]]),
                     landmark_extent=torch.tensor([[[.2,.2]]]),
                     landmark_valid=torch.ones(1,1,dtype=torch.bool),
                     landmark_text_mask=torch.ones(1,1,4,dtype=torch.bool))
        finally:
            hook.remove()
        self.assertEqual(len(calls),1)


if __name__ == '__main__':
    unittest.main()
