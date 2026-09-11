import unittest

import numpy as np
import torch

from multiagent.models.ET_haa import RegionGrounding
from multiagent.observation.cropclient import project_colrow_to_crop, region_target_from_crop


class CropProjectionTest(unittest.TestCase):
    def test_identity_projection_and_visibility(self):
        corners = np.array([(0, 0), (99, 0), (99, 99), (0, 99)], dtype=np.float32)
        center, inside = project_colrow_to_crop(corners, (49.5, 49.5), (100, 100))
        np.testing.assert_allclose(center, [49.5, 49.5], atol=1e-4)
        self.assertTrue(inside)
        _, outside = project_colrow_to_crop(corners, (120, 50), (100, 100))
        self.assertFalse(outside)

    def test_patch_label_and_outside_class(self):
        self.assertEqual(region_target_from_crop((112, 112), True, (224, 224)), 24)
        self.assertEqual(region_target_from_crop((0, 0), True, (224, 224)), 0)
        self.assertEqual(region_target_from_crop((0, 0), False, (224, 224)), 49)


class RegionGroundingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = RegionGrounding(visual_dim=8, d_model=12)
        self.frames = torch.randn(2, 1, 8, 49, requires_grad=True)
        self.language = torch.randn(2, 5, 12, requires_grad=True)
        self.mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)

    def test_retains_all_regions_and_adds_outside_class(self):
        regions = self.module.embed(self.frames)
        logits, pooled = self.module.ground(regions, self.language, self.mask)
        self.assertEqual(regions.shape, (2, 49, 12))
        self.assertEqual(logits.shape, (2, 50))
        self.assertEqual(pooled.shape, (2, 12))

    def test_outputs_depend_on_both_language_and_visual_regions(self):
        regions = self.module.embed(self.frames)
        logits, _ = self.module.ground(regions, self.language, self.mask)
        visual_grad, language_grad = torch.autograd.grad(
            logits.square().mean(), (self.frames, self.language)
        )
        self.assertGreater(visual_grad.norm().item(), 0)
        self.assertGreater(language_grad.norm().item(), 0)

    def test_language_shuffle_changes_region_scores(self):
        regions = self.module.embed(self.frames)
        original, _ = self.module.ground(regions, self.language, self.mask)
        shuffled, _ = self.module.ground(regions, self.language.flip(0), self.mask.flip(0))
        self.assertGreater((original[:, :49] - shuffled[:, :49]).abs().max().item(), 0)

    def test_masked_language_padding_has_no_effect(self):
        regions = self.module.embed(self.frames)
        original, _ = self.module.ground(regions, self.language, self.mask)
        changed = self.language.detach().clone()
        changed[0, 3:] += 1000
        modified, _ = self.module.ground(regions, changed, self.mask)
        torch.testing.assert_close(original[0], modified[0])

    def test_rejects_hidden_or_multi_frame_input(self):
        with self.assertRaises(ValueError):
            self.module.embed(torch.randn(2, 2, 8, 49))

    def test_visual_shuffle_changes_region_scores(self):
        regions = self.module.embed(self.frames)
        original, _ = self.module.ground(regions, self.language, self.mask)
        shuffled_regions = self.module.embed(self.frames.flip(0))
        shuffled, _ = self.module.ground(shuffled_regions, self.language, self.mask)
        self.assertGreater((original[:, :49] - shuffled[:, :49]).abs().max().item(), 0)


if __name__ == '__main__':
    unittest.main()
