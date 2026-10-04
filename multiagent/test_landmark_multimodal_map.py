import tempfile
import unittest
from pathlib import Path

import numpy as np

from multiagent.landmark_multimodal_map import (
    LandmarkMultimodalMap,
    LandmarkRecord,
)


class LandmarkMultimodalMapTest(unittest.TestCase):
    def setUp(self):
        self.record = LandmarkRecord(
            map_name="cambridge_block_3",
            landmark_id=7,
            name="Merton Hall",
            center_xy=(100.0, 200.0),
            normalized_xy=(0.25, 0.75),
            polygon=[(90.0, 190.0), (110.0, 210.0)],
            top_rgb_path="assets/top.jpg",
            ego_views=[{"rgb_path": "assets/ego.jpg", "camera_yaw": 0.3}],
        )
        self.semantic_map = LandmarkMultimodalMap([self.record])

    def test_cache_round_trip_keeps_visual_links(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            self.semantic_map.save(path)
            loaded = LandmarkMultimodalMap.load(path)
        record = loaded.find("cambridge_block_3", 7)
        self.assertEqual(record.name, "Merton Hall")
        self.assertEqual(record.top_rgb_path, "assets/top.jpg")
        self.assertEqual(record.ego_views[0]["rgb_path"], "assets/ego.jpg")

    def test_text_retrieval_and_scatter(self):
        retrieved = self.semantic_map.retrieve_text(
            "Fly toward Merton Hall and stop nearby.",
            "cambridge_block_3",
            topk=5,
        )
        self.assertEqual(retrieved[0][0].landmark_id, 7)

        prior = self.semantic_map.scatter_prior(retrieved, grid_size=7)
        self.assertEqual(prior.shape, (49,))
        self.assertAlmostEqual(float(prior.max()), 1.0, places=6)
        self.assertGreater(float(prior.sum()), 1.0)

    def test_unmatched_instruction_is_neutral(self):
        prior, retrieved = self.semantic_map.build_batch_prior(
            ["Fly to the white tower"],
            ["cambridge_block_3"],
            grid_size=7,
            topk=5,
        )
        self.assertEqual(retrieved, [[]])
        np.testing.assert_array_equal(prior, np.zeros((1, 49), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
