import unittest

import numpy as np
import torch

from pathlib import Path

from multiagent.static_belief_dataset import (
    StaticBeliefDataset, StaticBeliefSample, load_static_samples,
)


class _Rasterizer:
    def global_mask(self, map_name):
        return np.ones((16, 16), np.uint8)

    def referenced_mask(self, map_name, names):
        return np.full((16, 16), 2, np.uint8)


class StaticBeliefTrainingTest(unittest.TestCase):
    def setUp(self):
        self.sample = StaticBeliefSample("go", "map", ("church",), (8, 9), (4, 5), 0.0)
        self.language = {"go": torch.ones(1, 8)}

    def test_instruction_variant_has_no_map_or_pose_information(self):
        item = StaticBeliefDataset([self.sample], _Rasterizer(), self.language, "instruction", 16)[0]
        self.assertEqual(float(item["static_map"].abs().sum()), 0.0)

    def test_static_variants_have_strict_channel_separation(self):
        global_item = StaticBeliefDataset([self.sample], _Rasterizer(), self.language, "global", 16)[0]["static_map"]
        referenced = StaticBeliefDataset([self.sample], _Rasterizer(), self.language, "referenced", 16)[0]["static_map"]
        started = StaticBeliefDataset([self.sample], _Rasterizer(), self.language, "start_pose", 16)[0]["static_map"]
        self.assertGreater(float(global_item[0].sum()), 0)
        self.assertEqual(float(global_item[1:].sum()), 0.0)
        self.assertGreater(float(referenced[1].sum()), 0)
        self.assertEqual(float(referenced[2:].sum()), 0.0)
        self.assertEqual(int(started[2].sum()), 1)
        self.assertTrue(torch.all(started[3] == 1))
        self.assertTrue(torch.all(started[4] == 0))

    def test_development_loader_rejects_test_unseen(self):
        with self.assertRaisesRegex(ValueError, "test_unseen"):
            load_static_samples(Path("."), "test_unseen")


if __name__ == "__main__":
    unittest.main()
