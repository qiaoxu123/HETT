import unittest

import numpy as np

from multiagent.maps.landmark_map import LandmarkMap
from scripts.evaluate_target_contrasts import pair_manifest, select_pairs


class TargetContrastSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objects, cls.pairs = select_pairs('val_unseen', 8)

    def test_pairs_hold_map_and_landmark_prior_constant(self):
        for left, right, landmarks in self.pairs:
            self.assertEqual(left.map_name, right.map_name)
            self.assertNotEqual(left.object_id, right.object_id)
            self.assertEqual(landmarks, tuple(sorted(
                self.objects[left.map_name][left.object_id]
                .processed_descriptions[left.desc_id].landmarks
            )))
            self.assertEqual(landmarks, tuple(sorted(
                self.objects[right.map_name][right.object_id]
                .processed_descriptions[right.desc_id].landmarks
            )))

    def test_targets_are_separated_and_maps_are_balanced(self):
        rows = pair_manifest(self.objects, self.pairs)
        self.assertTrue(all(row['target_distance_m'] >= 20 for row in rows))
        self.assertGreaterEqual(len({row['map'] for row in rows}), 3)

    def test_landmark_rasters_are_identical_within_each_pair(self):
        for left, right, _ in self.pairs:
            left_names = self.objects[left.map_name][left.object_id].processed_descriptions[left.desc_id].landmarks
            right_names = self.objects[right.map_name][right.object_id].processed_descriptions[right.desc_id].landmarks
            left_map = LandmarkMap(left.map_name, (240, 240), 240 / 410, left_names).to_array()
            right_map = LandmarkMap(right.map_name, (240, 240), 240 / 410, right_names).to_array()
            self.assertTrue(np.array_equal(left_map, right_map))


if __name__ == '__main__':
    unittest.main()
