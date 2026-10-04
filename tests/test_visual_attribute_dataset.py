import unittest

from multiagent.visual_attributes.parser import parse_attributes
from multiagent.visual_attributes.taxonomy import GEOMETRIC_CATEGORIES, VISUAL_CATEGORIES


class VisualAttributeDatasetTest(unittest.TestCase):
    def test_parser_separates_visual_and_geometric_constraints(self):
        parsed = parse_attributes("The small white rectangular building on the left beside the main road")
        self.assertIn("white", parsed["color"])
        self.assertIn("small", parsed["size"])
        self.assertIn("rectangular", parsed["shape"])
        self.assertIn("road", parsed["context"])
        self.assertIn("left", parsed["spatial_relation"])
        self.assertTrue(set(VISUAL_CATEGORIES).isdisjoint(GEOMETRIC_CATEGORIES))

    def test_lexical_boundaries_avoid_substring_matches(self):
        self.assertNotIn("color", parse_attributes("Fred is nearby"))

    def test_test_unseen_census_is_forbidden(self):
        from analysis.visual_attribute_census import census
        with self.assertRaisesRegex(ValueError, "test_unseen"):
            census(None, ("test_unseen",))


if __name__ == "__main__": unittest.main()

