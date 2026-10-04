import unittest

from multiagent.visual_attributes.dataset import leakage_report


class VisualAttributeLeakageTest(unittest.TestCase):
    def test_object_and_episode_grouping(self):
        rows = [
            {"split": "train_seen", "object_key": "a:1", "episode_ids": ["train:1"]},
            {"split": "val_seen", "object_key": "a:2", "episode_ids": ["val:1"]},
            {"split": "val_unseen", "object_key": "b:1", "episode_ids": ["unseen:1"]},
        ]
        report = leakage_report(rows)
        self.assertFalse(report["forbidden_test_unseen"])
        self.assertTrue(all(value == 0 for value in report["object_overlap"].values()))
        self.assertTrue(all(value == 0 for value in report["episode_overlap"].values()))

    def test_overlap_is_detected(self):
        rows = [
            {"split": "train_seen", "object_key": "a:1", "episode_ids": ["same"]},
            {"split": "val_seen", "object_key": "a:1", "episode_ids": ["same"]},
        ]
        report = leakage_report(rows)
        self.assertEqual(report["object_overlap"]["train_seen/val_seen"], 1)
        self.assertEqual(report["episode_overlap"]["train_seen/val_seen"], 1)


if __name__ == "__main__": unittest.main()

