import unittest

import torch

from multiagent.models.goal_predictor import MapEncoder


class GlobalLandmarkPriorTest(unittest.TestCase):
    def test_four_channel_map_encoder_shape(self):
        encoder = MapEncoder(240, input_channels=4)
        x = torch.zeros(2, 4, 240, 240)
        y = encoder(x)
        self.assertEqual(y.shape, (2, encoder.out_features))

    def test_global_and_referenced_channels_are_independent_inputs(self):
        encoder = MapEncoder(240, input_channels=4)
        first_conv = encoder.main[1]
        self.assertEqual(first_conv.in_channels, 4)
        self.assertEqual(first_conv.weight.shape[1], 4)


if __name__ == "__main__":
    unittest.main()
