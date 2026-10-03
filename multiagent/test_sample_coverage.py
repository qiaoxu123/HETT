from collections import Counter
from types import SimpleNamespace
import unittest

from multiagent.env import CityNavBatch


class SampleCoverageTest(unittest.TestCase):
    def test_full_epoch_visits_all_samples_with_only_tail_padding(self):
        dataset = CityNavBatch.__new__(CityNavBatch)
        dataset.batch_size = 16
        dataset.data = [
            SimpleNamespace(id=i, map_name="map", target_description="target")
            for i in range(21877)
        ]

        visits = Counter()
        batch_count = 0
        for _ in dataset.next_batch():
            batch_count += 1
            visits.update(item.id for item in dataset.batch)

        self.assertEqual(batch_count, 1368)
        self.assertEqual(len(visits), 21877)
        self.assertEqual(sum(visits.values()), 1368 * 16)
        self.assertEqual([i for i, count in visits.items() if count == 2], list(range(11)))
        self.assertTrue(all(count in (1, 2) for count in visits.values()))


if __name__ == "__main__":
    unittest.main()
