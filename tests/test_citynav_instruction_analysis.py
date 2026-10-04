import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_citynav_instructions import (
    SPLITS,
    alphanumeric,
    create_report,
    load_records,
    marker_names,
)


class CityNavInstructionAnalysisTest(unittest.TestCase):
    def test_uses_model_input_description_and_preserves_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cityrefer = root / "cityrefer"
            trajectories = root / "processed_citynav"
            cityrefer.mkdir()
            trajectories.mkdir()
            (cityrefer / "objects.json").write_text(json.dumps({
                "city_block_1": {"7": {
                    "object_type": "Building",
                    "descriptions": ["The blue building on the left."],
                }}
            }), encoding="utf-8")
            (cityrefer / "processed_descriptions.json").write_text(json.dumps({
                "city_block_1": {"7": [{
                    "target": "building", "landmarks": ["tower"], "surroundings": [],
                }]}
            }), encoding="utf-8")
            for split in SPLITS:
                records = [{
                    "area": "city", "block": 1, "object_ids": [7], "ann_ids": [0],
                    "descriptions": ["The blue building on the left!"],
                }]
                (trajectories / f"citynav_{split}.json").write_text(
                    json.dumps(records), encoding="utf-8"
                )

            rows, sources = load_records(root)
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["text"], "The blue building on the left.")
            self.assertFalse(rows[0]["raw_matches"])
            self.assertEqual(rows[0]["landmarks"], 1)
            self.assertIn("left/right", rows[0]["markers"])
            self.assertEqual(
                alphanumeric(rows[0]["text"]), alphanumeric(rows[0]["raw_text"])
            )
            report = create_report(rows, sources, root)
            self.assertIn("train_seen\t0\tcity_block_1:7:0", report)
            self.assertIn('"The blue building on the left."', report)

    def test_markers_use_word_boundaries(self):
        self.assertIn("between", marker_names("The car between two buildings"))
        self.assertNotIn("left/right", marker_names("leftover paint"))


if __name__ == "__main__":
    unittest.main()
