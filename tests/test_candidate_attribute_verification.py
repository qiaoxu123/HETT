import unittest

from scripts.evaluate_candidate_verification import candidate_world, required_attributes


class CandidateAttributeVerificationTest(unittest.TestCase):
    def test_parser_separates_visual_and_geometry_compatible_labels(self):
        result = required_attributes(
            "Fly to the large white rectangular building next to the road with a flat roof."
        )
        self.assertEqual(result["color"], "white")
        self.assertEqual(result["size"], "large")
        self.assertEqual(result["shape"], "rectangular")
        self.assertEqual(result["semantic"], "Building")
        self.assertEqual(result["road_context"], "yes")
        self.assertEqual(result["roof_presence"], "yes")

    def test_grid_to_world_is_cell_centered(self):
        x0, y0 = candidate_world("cambridge_block_2", 0, 0)
        x1, y1 = candidate_world("cambridge_block_2", 1, 1)
        self.assertAlmostEqual(x1 - x0, 410 / 30)
        self.assertAlmostEqual(y0 - y1, 410 / 30)


if __name__ == "__main__":
    unittest.main()
