import unittest
from pathlib import Path
import tempfile

from scripts.analyze_multiseed import aggregate, plot, statistics


def run(successes, sr):
    episodes = {
        str(index): {'success': success, 'final_distance': 0 if success else 30,
                     'map': f'map{index % 2}'}
        for index, success in enumerate(successes)
    }
    return {'val_unseen': {'reported': {'sr': sr, 'spl': sr / 2, 'ne': 20},
                           'episodes': episodes}}


class MultiSeedAnalysisTest(unittest.TestCase):
    def test_statistics_keeps_each_seed_and_uses_sample_deviation(self):
        result = statistics([10, 20, 30])
        self.assertEqual(result['n'], 3)
        self.assertEqual(result['mean'], 20)
        self.assertEqual(result['std'], 10)
        self.assertEqual(result['values'], [10.0, 20.0, 30.0])
        self.assertLess(result['ci95'][0], 20)
        self.assertGreater(result['ci95'][1], 20)

    def test_aggregate_is_paired_within_seed(self):
        loaded = {
            'base': {0: run([0, 1, 0, 1], 50), 17: run([0, 0, 1, 1], 50),
                     42: run([0, 1, 0, 1], 50)},
            'new': {0: run([1, 1, 0, 1], 75), 17: run([1, 0, 1, 1], 75),
                    42: run([1, 1, 0, 1], 75)},
        }
        result = aggregate(loaded, [('base', 'new')])
        self.assertEqual(result['runs']['new']['val_unseen']['sr']['n'], 3)
        paired = result['pairs']['base:new']['val_unseen']
        self.assertEqual(paired['success_delta_pp']['mean'], 25)
        self.assertEqual(set(paired['per_seed']), {'0', '17', '42'})
        self.assertEqual(paired['success_wins_total'], 3)

        with tempfile.TemporaryDirectory() as directory:
            plot(result, Path(directory))
            self.assertGreater((Path(directory) / 'multiseed_metrics.png').stat().st_size, 0)
            self.assertGreater((Path(directory) / 'multiseed_paired_deltas.png').stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
