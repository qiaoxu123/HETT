"""SBF-inspired candidate-relation evidence unit tests (CPU only)."""
import unittest

import torch

from multiagent.models.candidate_relation_selector import (
    CandidateRelationSelector, candidate_anchor_geometry,
    candidate_pose_history_features,
)
from multiagent.models.heatmap_trajectory import HeatmapTrajectoryHead


class CandidateRelationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.selector = CandidateRelationSelector(
            feature_dim=16, language_dim=12, hidden_dim=24,
            attention_heads=4, dropout=0.).eval()
        self.goals = torch.tensor([
            [[.2, .2], [.8, .6], [.5, .1]],
            [[.4, .5], [.7, .1], [.6, .8]],
        ])
        self.features = torch.randn(2, 3, 16)
        self.current = torch.tensor([[.1, .2], [.2, .2]])
        self.heading = torch.tensor([[0., 1.], [1., 0.]])
        self.language = torch.randn(2, 5, 12)
        self.language_valid = torch.tensor(
            [[True, True, True, True, False],
             [True, True, True, False, False]])
        self.landmark = torch.tensor([
            [[.15, .35], [.7, .6]],
            [[.5, .4], [0., 0.]],
        ])
        self.extent = torch.ones_like(self.landmark) * .1
        self.valid = torch.tensor([[True, True], [True, False]])
        self.spans = torch.zeros(2, 2, 5, dtype=torch.bool)
        self.spans[0, 0, 1] = True
        self.spans[0, 1, 3] = True
        self.spans[1, 0, 2] = True
        self.history = torch.tensor([
            [[.0, .1], [.1, .2]],
            [[.15, .1], [.2, .2]],
        ])

    def run_selector(self, **kwargs):
        inputs = dict(goal_xy=self.goals, goal_features=self.features,
                      current_xy=self.current, heading_sc=self.heading,
                      language_tokens=self.language,
                      language_mask=self.language_valid,
                      landmark_xy=self.landmark,
                      landmark_extent=self.extent,
                      landmark_valid=self.valid,
                      landmark_text_mask=self.spans,
                      history_xy=self.history)
        inputs.update(kwargs)
        return self.selector(**inputs)

    def test_xy_world_north_geometry(self):
        result = candidate_anchor_geometry(
            torch.tensor([[[.7, .2]]]),
            torch.tensor([[[.5, .5]]]),
            torch.tensor([[[.2, .2]]]))
        self.assertAlmostEqual(float(result[0, 0, 0, 0]), .2, places=5)
        self.assertAlmostEqual(float(result[0, 0, 0, 1]), .3, places=5)
        self.assertGreater(float(result[0, 0, 0, 4]), 0.)

    def test_anchor_identity_pairs_are_permutation_invariant(self):
        original = self.run_selector()
        perm = [1, 0]
        reordered = self.run_selector(
            landmark_xy=self.landmark[:, perm],
            landmark_extent=self.extent[:, perm],
            landmark_valid=self.valid[:, perm],
            landmark_text_mask=self.spans[:, perm])
        self.assertEqual(tuple(original.shape), (2, 3))
        self.assertTrue(torch.allclose(original, reordered, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.isfinite(original).all())

    def test_wrong_name_to_anchor_pair_changes_candidate_evidence(self):
        original = self.run_selector()
        wrong = self.spans.clone()
        wrong[0] = wrong[0, [1, 0]]
        altered = self.run_selector(landmark_text_mask=wrong)
        self.assertGreater(float((original[0] - altered[0]).abs().max()), 1e-7)

    def test_empty_landmark_rows_remain_exact_zero_and_finite(self):
        valid = self.valid.clone()
        valid[1] = False
        out = self.run_selector(landmark_valid=valid)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(out[1], torch.zeros_like(out[1])))
        all_invalid = self.run_selector(landmark_valid=torch.zeros_like(valid))
        self.assertTrue(torch.equal(all_invalid, torch.zeros_like(all_invalid)))

    def test_pose_heading_and_history_are_candidate_conditioned(self):
        geom = candidate_pose_history_features(
            self.goals, self.current, self.heading, self.history)
        self.assertEqual(tuple(geom.shape), (2, 3, 7))
        without_history = candidate_pose_history_features(
            self.goals, self.current, self.heading)
        self.assertTrue(torch.equal(without_history[..., -2:],
                                    torch.zeros_like(without_history[..., -2:])))
        original = self.run_selector()
        shifted = self.run_selector(
            history_xy=self.history + .2,
            heading_sc=torch.tensor([[1., 0.], [0., 1.]]))
        self.assertGreater(float((original - shifted).abs().max()), 1e-7)

    def test_selector_gradient_flows_from_candidate_score(self):
        feature = self.features.detach().clone().requires_grad_(True)
        out = self.run_selector(goal_features=feature)
        loss = out[0, 0] - out[0, 1]
        loss.backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertGreater(float(feature.grad.abs().sum()), 0.)
        self.assertIsNotNone(self.selector.geometry_proj[0].weight.grad)

    def test_relation_gate_preserves_prior_at_initialization_and_can_change_ranking(self):
        head = HeatmapTrajectoryHead(
            feature_dim=16, language_dim=12, hidden_dim=32,
            modes=3, waypoints=6).eval()
        field = torch.randn(2, 16, 9, 9)
        belief = torch.softmax(torch.randn(2, 81), -1).reshape(2, 9, 9)
        arguments = dict(top_k=3, language_tokens=self.language,
                         language_mask=self.language_valid,
                         landmark_xy=self.landmark,
                         landmark_extent=self.extent,
                         landmark_valid=self.valid,
                         landmark_text_mask=self.spans,
                         history_xy=self.history)
        with torch.no_grad():
            original, _ = head(field, belief, self.current, self.heading,
                               relation_enabled=False, **arguments)
            zero_gated, _ = head(field, belief, self.current, self.heading,
                                 relation_enabled=True, **arguments)
            self.assertTrue(torch.allclose(original.joint_logits,
                                            zero_gated.joint_logits, atol=1e-6))
            head.relation_gate.fill_(1.)
            active, _ = head(field, belief, self.current, self.heading,
                             relation_enabled=True, **arguments)
            self.assertTrue(torch.isfinite(active.joint_logits).all())
            self.assertGreater(
                float((active.joint_logits - original.joint_logits).abs().max()),
                1e-7)


if __name__ == "__main__":
    unittest.main()
