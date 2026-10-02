import unittest

import torch

from multiagent.models.goal_predictor import MapEncoder


class GlobalLandmarkPriorTest(unittest.TestCase):
    def test_baseline_hett_map_stays_three_channel(self):
        encoder = MapEncoder(240, input_channels=3)
        x = torch.zeros(2, 3, 240, 240)
        y = encoder(x)
        self.assertEqual(y.shape, (2, encoder.out_features))
        self.assertEqual(encoder.main[1].in_channels, 3)

    def test_global_prior_has_independent_one_channel_encoder(self):
        encoder = MapEncoder(240, input_channels=1)
        x = torch.zeros(2, 1, 240, 240)
        y = encoder(x)
        self.assertEqual(y.shape, (2, encoder.out_features))
        self.assertEqual(encoder.main[1].in_channels, 1)


if __name__ == "__main__":
    unittest.main()
