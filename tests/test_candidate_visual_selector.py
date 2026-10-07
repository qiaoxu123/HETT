import inspect
import unittest

import torch

from multiagent.models.candidate_selector import (
    CandidateVisualSelector,
    candidate_ranking_loss,
    heatmap_ids_to_normalized_xy,
    relative_geometry,
    sample_candidate_features_from_current_view,
    select_candidate_with_abstention,
)


class CandidateVisualSelectorTest(unittest.TestCase):
    def test_heatmap_ids_use_col_as_x_and_row_as_y(self):
        ids = torch.tensor([[2 * 10 + 7]])
        xy = heatmap_ids_to_normalized_xy(ids, field_size=10)
        self.assertTrue(torch.allclose(xy, torch.tensor([[[0.75, 0.25]]])))

    def test_current_view_sampling_respects_yaw_and_visibility(self):
        # Feature values make the center and top-center cells easy to identify.
        feature = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
        agent_xy = torch.tensor([[0.5, 0.5]])
        # yaw=0 points along world +x. A candidate 10 m east therefore appears
        # at image top when view radius is 10 m.
        candidates = torch.tensor([[
            [0.5, 0.5],
            [0.6, 0.5],
            [0.8, 0.5],
        ]])
        sampled, visible = sample_candidate_features_from_current_view(
            feature,
            candidates,
            agent_xy,
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([10.0]),
            map_meters=100.0,
        )
        self.assertTrue(torch.equal(visible, torch.tensor([[True, True, False]])))
        self.assertAlmostEqual(float(sampled[0, 0, 0]), 4.0, places=5)
        self.assertAlmostEqual(float(sampled[0, 1, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(sampled[0, 2, 0]), 0.0, places=5)

    def test_relative_geometry_matches_sbf_five_tuple(self):
        origin = torch.tensor([[[0.0, 0.0]]])
        destination = torch.tensor([[[3.0, 4.0]]])
        geometry = relative_geometry(origin, destination)
        expected = torch.tensor([[[3.0, 4.0, 5.0, 0.8, 0.6]]])
        self.assertTrue(torch.allclose(geometry, expected, atol=1e-6))

    def test_selector_is_candidate_permutation_equivariant(self):
        torch.manual_seed(0)
        model = CandidateVisualSelector(
            visual_dim=8,
            language_dim=8,
            hidden_dim=8,
            layers=2,
            attention_heads=2,
            dropout=0.0,
        ).eval()
        visual = torch.randn(1, 3, 8)
        language = torch.randn(1, 5, 8)
        language_mask = torch.ones(1, 5, dtype=torch.bool)
        candidate_xy = torch.tensor([[[0.2, 0.2], [0.5, 0.4], [0.8, 0.7]]])
        agent_xy = torch.tensor([[0.1, 0.1]])
        landmark_features = torch.randn(1, 2, 8)
        landmark_xy = torch.tensor([[[0.3, 0.3], [0.7, 0.6]]])
        landmark_mask = torch.tensor([[True, True]])

        logits = model(
            visual,
            language,
            candidate_xy,
            agent_xy,
            landmark_features,
            landmark_xy,
            landmark_mask,
            language_mask,
        )
        permutation = torch.tensor([2, 0, 1])
        permuted = model(
            visual[:, permutation],
            language,
            candidate_xy[:, permutation],
            agent_xy,
            landmark_features,
            landmark_xy,
            landmark_mask,
            language_mask,
        )
        self.assertTrue(torch.allclose(permuted, logits[:, permutation], atol=1e-6))

    def test_selector_is_landmark_permutation_invariant(self):
        torch.manual_seed(1)
        model = CandidateVisualSelector(
            visual_dim=8,
            language_dim=8,
            hidden_dim=8,
            layers=1,
            attention_heads=2,
            dropout=0.0,
        ).eval()
        visual = torch.randn(1, 2, 8)
        language = torch.randn(1, 4, 8)
        language_mask = torch.ones(1, 4, dtype=torch.bool)
        candidate_xy = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]])
        agent_xy = torch.tensor([[0.5, 0.5]])
        landmark_features = torch.randn(1, 2, 8)
        landmark_xy = torch.tensor([[[0.1, 0.2], [0.9, 0.7]]])
        landmark_mask = torch.tensor([[True, True]])

        logits = model(
            visual, language, candidate_xy, agent_xy,
            landmark_features, landmark_xy, landmark_mask, language_mask,
        )
        order = torch.tensor([1, 0])
        permuted = model(
            visual, language, candidate_xy, agent_xy,
            landmark_features[:, order], landmark_xy[:, order],
            landmark_mask[:, order], language_mask,
        )
        self.assertTrue(torch.allclose(permuted, logits, atol=1e-6))

    def test_geometry_ablation_removes_coordinate_dependence(self):
        torch.manual_seed(2)
        model = CandidateVisualSelector(
            visual_dim=8,
            language_dim=8,
            hidden_dim=8,
            layers=1,
            attention_heads=2,
            dropout=0.0,
            use_geometry=False,
        ).eval()
        visual = torch.randn(1, 2, 8)
        language = torch.randn(1, 4, 8)
        language_mask = torch.ones(1, 4, dtype=torch.bool)
        agent_xy = torch.tensor([[0.5, 0.5]])
        landmark_features = torch.randn(1, 1, 8)
        landmark_mask = torch.tensor([[True]])

        first = model(
            visual, language,
            torch.tensor([[[0.1, 0.1], [0.2, 0.2]]]),
            agent_xy, landmark_features,
            torch.tensor([[[0.3, 0.3]]]),
            landmark_mask, language_mask,
        )
        second = model(
            visual, language,
            torch.tensor([[[0.8, 0.8], [0.9, 0.9]]]),
            agent_xy, landmark_features,
            torch.tensor([[[0.7, 0.7]]]),
            landmark_mask, language_mask,
        )
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_selector_api_has_no_heatmap_score_input(self):
        parameters = set(inspect.signature(CandidateVisualSelector.forward).parameters)
        self.assertNotIn("field_score", parameters)
        self.assertNotIn("heatmap_score", parameters)
        self.assertNotIn("reference_prior", parameters)

    def test_abstains_with_insufficient_visible_candidates(self):
        decision = select_candidate_with_abstention(
            torch.tensor([11]),
            torch.tensor([[11, 22, 33]]),
            torch.tensor([[0.0, 5.0, -2.0]]),
            torch.tensor([[False, True, False]]),
            min_visible_candidates=2,
            min_confidence=0.55,
            min_margin=0.10,
        )
        self.assertFalse(bool(decision.triggered[0]))
        self.assertEqual(int(decision.chosen_ids[0]), 11)

    def test_confident_visible_evidence_can_replace_heatmap_top1(self):
        decision = select_candidate_with_abstention(
            torch.tensor([11]),
            torch.tensor([[11, 22, 33]]),
            torch.tensor([[0.0, 4.0, -3.0]]),
            torch.tensor([[True, True, False]]),
            min_visible_candidates=2,
            min_confidence=0.55,
            min_margin=0.10,
        )
        self.assertTrue(bool(decision.triggered[0]))
        self.assertEqual(int(decision.chosen_ids[0]), 22)

    def test_low_margin_preserves_heatmap_top1(self):
        decision = select_candidate_with_abstention(
            torch.tensor([11]),
            torch.tensor([[11, 22, 33]]),
            torch.tensor([[1.00, 1.01, -3.0]]),
            torch.tensor([[True, True, False]]),
            min_visible_candidates=2,
            min_confidence=0.50,
            min_margin=0.10,
        )
        self.assertFalse(bool(decision.triggered[0]))
        self.assertEqual(int(decision.chosen_ids[0]), 11)

    def test_ranking_loss_requires_good_visible_nearest_candidate(self):
        logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        distances = torch.tensor([[35.0, 10.0, 60.0]])
        visible = torch.tensor([[True, True, False]])
        result = candidate_ranking_loss(
            logits,
            distances,
            visible,
            torch.tensor([True]),
            good_radius_m=20.0,
            min_visible_candidates=2,
        )
        self.assertEqual(int(result.eligible_count), 1)
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertIsNotNone(logits.grad)

    def test_ranking_loss_abstains_when_good_candidate_is_not_visible(self):
        logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        distances = torch.tensor([[35.0, 10.0, 60.0]])
        visible = torch.tensor([[True, False, True]])
        result = candidate_ranking_loss(
            logits,
            distances,
            visible,
            torch.tensor([True]),
            good_radius_m=20.0,
            min_visible_candidates=2,
        )
        self.assertEqual(int(result.eligible_count), 0)
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
