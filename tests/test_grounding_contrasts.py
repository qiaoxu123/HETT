import unittest

from scripts.evaluate_grounding_contrasts import pair_manifest, select_pairs


class GroundingContrastSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objects, cls.pairs = select_pairs('val_unseen', 8)

    def test_pairs_hold_landmark_prior_constant(self):
        for left, right, landmarks in self.pairs:
            self.assertEqual(left.map_name, right.map_name)
            self.assertNotEqual(left.object_id, right.object_id)
            self.assertEqual(landmarks,
                             tuple(sorted(self.objects[left.map_name][left.object_id]
                                          .processed_descriptions[left.desc_id].landmarks)))
            self.assertEqual(landmarks,
                             tuple(sorted(self.objects[right.map_name][right.object_id]
                                          .processed_descriptions[right.desc_id].landmarks)))

    def test_targets_are_distinct_and_maps_are_balanced(self):
        rows = pair_manifest(self.objects, self.pairs)
        self.assertTrue(all(row['target_distance_m'] >= 20 for row in rows))
        self.assertGreaterEqual(len({row['map'] for row in rows}), 3)


if __name__ == '__main__':
    unittest.main()
