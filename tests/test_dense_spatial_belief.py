import unittest

import torch

from multiagent.models.spatial_belief import (
    CompactSpatialBelief,
    greedy_nms_topk,
    local_soft_argmax_xy,
    metric_gaussian_target,
    should_apply_reference_rerank,
    referenced_landmark_proximity_prior,
    rerank_heatmap_topk_with_reference,
)


class DenseSpatialBeliefTest(unittest.TestCase):
    def test_output_shape_and_probability_normalization(self):
        model = CompactSpatialBelief(
            input_channels=4,
            field_size=28,
            hidden_dim=32,
            language_dim=32,
            attention_heads=4,
            dropout=0.0,
        )
        maps = torch.zeros(2, 4, 240, 240)
        language = torch.randn(2, 6, 32)
        mask = torch.ones(2, 6, dtype=torch.bool)
        output = model(maps, language, mask)
        self.assertEqual(tuple(output.logits.shape), (2, 28, 28))
        self.assertTrue(torch.allclose(
            output.probabilities.sum(dim=(1, 2)),
            torch.ones(2),
            atol=1e-5,
        ))

    def test_metric_gaussian_is_normalized(self):
        target = metric_gaussian_target(
            torch.tensor([[0.5, 0.5]]),
            field_size=28,
            sigma_m=20.0,
            map_meters=410.0,
        )
        self.assertEqual(tuple(target.shape), (1, 28, 28))
        self.assertAlmostEqual(float(target.sum()), 1.0, places=5)


    def test_metric_gaussian_respects_xy_to_row_col_convention(self):
        field_size = 10
        goal_xy = torch.tensor([[0.75, 0.25]])
        target = metric_gaussian_target(
            goal_xy,
            field_size=field_size,
            sigma_m=1.0,
            map_meters=100.0,
        )
        peak_id = int(target.flatten(1).argmax(dim=1).item())
        peak_row = peak_id // field_size
        peak_col = peak_id % field_size

        # goal=(x=.75,y=.25) must map near row=y*H, col=x*W.
        self.assertEqual(peak_row, 2)
        self.assertEqual(peak_col, 7)

    def test_belief_cell_to_normalized_xy_uses_col_then_row(self):
        field_size = 10
        row = torch.tensor([2.0])
        col = torch.tensor([7.0])
        xy = torch.stack(
            (
                (col + 0.5) / field_size,
                (row + 0.5) / field_size,
            ),
            dim=1,
        )
        self.assertTrue(torch.allclose(xy, torch.tensor([[0.75, 0.25]])))


    def test_local_soft_argmax_refines_inside_selected_mode(self):
        belief = torch.zeros(1, 5, 5)
        belief[0, 1, 3] = 0.6
        belief[0, 1, 4] = 0.3
        belief[0, 2, 3] = 0.1
        peak_id = torch.tensor([1 * 5 + 3])
        xy = local_soft_argmax_xy(belief, peak_id, window_size=3)

        # The refined point should stay near the selected peak, but shift
        # toward the neighboring probability mass instead of using the
        # discrete cell center exactly.
        center_xy = torch.tensor([[(3.5 / 5), (1.5 / 5)]])
        self.assertGreater(float(xy[0, 0]), float(center_xy[0, 0]))
        self.assertGreater(float(xy[0, 1]), float(center_xy[0, 1]))
        self.assertLess(float(xy[0, 0]), 1.0)
        self.assertLess(float(xy[0, 1]), 1.0)

    def test_greedy_nms_suppresses_neighboring_peak(self):
        belief = torch.zeros(1, 7, 7)
        belief[0, 1, 1] = 0.50
        belief[0, 1, 2] = 0.49
        belief[0, 5, 5] = 0.40
        ids = greedy_nms_topk(belief, top_k=2, kernel_size=3)
        self.assertEqual(ids[0, 0].item(), 1 * 7 + 1)
        self.assertEqual(ids[0, 1].item(), 5 * 7 + 5)



    def test_reference_rerank_is_disabled_during_student_training_rollout(self):
        self.assertFalse(should_apply_reference_rerank(
            feedback="student", train_ml=0.2, disabled=False
        ))

    def test_reference_rerank_is_enabled_only_for_student_inference(self):
        self.assertTrue(should_apply_reference_rerank(
            feedback="student", train_ml=None, disabled=False
        ))
        self.assertFalse(should_apply_reference_rerank(
            feedback="teacher", train_ml=None, disabled=False
        ))
        self.assertFalse(should_apply_reference_rerank(
            feedback="student", train_ml=None, disabled=True
        ))

    def test_reference_prior_expands_without_target_information(self):
        maps = torch.zeros(1, 4, 9, 9)
        maps[0, 3, 4, 4] = 1.0
        prior = referenced_landmark_proximity_prior(
            maps, field_size=9, dilation_steps=2, decay=0.5
        )
        self.assertAlmostEqual(float(prior[0, 4, 4]), 1.0)
        self.assertAlmostEqual(float(prior[0, 4, 5]), 0.5)
        self.assertAlmostEqual(float(prior[0, 4, 6]), 0.25)
        self.assertEqual(float(prior[0, 0, 0]), 0.0)

    def test_reference_rerank_changes_only_ambiguous_topk(self):
        probs = torch.full((1, 5, 5), 1e-4)
        probs[0, 1, 1] = 0.40
        probs[0, 3, 3] = 0.35
        probs = probs / probs.sum(dim=(1, 2), keepdim=True)
        topk = torch.tensor([[1 * 5 + 1, 3 * 5 + 3]])
        prior = torch.zeros_like(probs)
        prior[0, 3, 3] = 1.0
        selected, changed = rerank_heatmap_topk_with_reference(
            probs,
            topk,
            prior,
            rerank_top_k=2,
            prior_weight=1.0,
            max_log_margin=0.5,
            min_prior_gain=0.2,
        )
        self.assertTrue(bool(changed[0]))
        self.assertEqual(int(selected[0]), 3 * 5 + 3)

    def test_reference_rerank_preserves_confident_top1(self):
        probs = torch.full((1, 5, 5), 1e-5)
        probs[0, 1, 1] = 0.90
        probs[0, 3, 3] = 0.05
        probs = probs / probs.sum(dim=(1, 2), keepdim=True)
        topk = torch.tensor([[1 * 5 + 1, 3 * 5 + 3]])
        prior = torch.zeros_like(probs)
        prior[0, 3, 3] = 1.0
        selected, changed = rerank_heatmap_topk_with_reference(
            probs,
            topk,
            prior,
            rerank_top_k=2,
            prior_weight=2.0,
            max_log_margin=0.35,
            min_prior_gain=0.2,
        )
        self.assertFalse(bool(changed[0]))
        self.assertEqual(int(selected[0]), 1 * 5 + 1)

    def test_reference_rerank_is_noop_without_reference_mask(self):
        probs = torch.zeros(1, 5, 5)
        probs[0, 1, 1] = 0.51
        probs[0, 3, 3] = 0.49
        topk = torch.tensor([[1 * 5 + 1, 3 * 5 + 3]])
        prior = torch.zeros_like(probs)
        selected, changed = rerank_heatmap_topk_with_reference(
            probs, topk, prior, rerank_top_k=2
        )
        self.assertFalse(bool(changed[0]))
        self.assertEqual(int(selected[0]), 1 * 5 + 1)


if __name__ == "__main__":
    unittest.main()
