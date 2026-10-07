import inspect
import unittest

import torch

from multiagent.models.candidate_selector import (
    CandidateVisualSelector,
    candidate_ranking_loss,
    heatmap_ids_to_normalized_xy,
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

    def test_selector_is_candidate_permutation_equivariant(self):
        torch.manual_seed(0)
        model = CandidateVisualSelector(
            visual_dim=8,
            language_dim=8,
            hidden_dim=8,
            attention_heads=2,
            dropout=0.0,
        ).eval()
        visual = torch.randn(1, 3, 8)
        language = torch.randn(1, 5, 8)
        mask = torch.ones(1, 5, dtype=torch.bool)

        logits = model(visual, language, mask)
        permutation = torch.tensor([2, 0, 1])
        permuted = model(visual[:, permutation], language, mask)
        self.assertTrue(torch.allclose(permuted, logits[:, permutation], atol=1e-6))

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
