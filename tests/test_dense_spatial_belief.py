import unittest

import torch

from multiagent.models.spatial_belief import (
    CompactSpatialBelief,
    greedy_nms_topk,
    metric_gaussian_target,
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

    def test_greedy_nms_suppresses_neighboring_peak(self):
        belief = torch.zeros(1, 7, 7)
        belief[0, 1, 1] = 0.50
        belief[0, 1, 2] = 0.49
        belief[0, 5, 5] = 0.40
        ids = greedy_nms_topk(belief, top_k=2, kernel_size=3)
        self.assertEqual(ids[0, 0].item(), 1 * 7 + 1)
        self.assertEqual(ids[0, 1].item(), 5 * 7 + 5)


if __name__ == "__main__":
    unittest.main()
