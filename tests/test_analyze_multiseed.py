import unittest
from pathlib import Path
import tempfile

from scripts.analyze_multiseed import (
    aggregate, plot, plot_failure_rates, select_failure_cases, statistics,
)


def run(successes, sr):
    episodes = {
        str(index): {'success': success, 'final_distance': 0 if success else 30,
                     'best_distance': 0 if success else 25,
                     'any_success': bool(success), 'hit_then_lost': False,
                     'false_stop': not success, 'stopped': not success,
                     'actions': 4, 'stage1_actions': 3, 'stage2_actions': 1,
                     'switches': 1, 'recoveries': 0,
                     'id': str(index), 'map': f'map{index % 2}'}
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

    def test_failure_cases_keep_worst_stops_and_paired_rescues(self):
        loaded = {
            'base': {0: run([0, 1], 50)},
            'new': {0: run([1, 0], 50)},
        }
        result = select_failure_cases(loaded, [('base', 'new')], limit=10)
        base = result['runs']['base']['0']['val_unseen']
        self.assertEqual(base['worst_final_distance'][0]['final_distance'], 30)
        self.assertEqual(len(base['false_stops']), 1)
        pair = result['pairs']['base:new']['0']['val_unseen']
        self.assertEqual(len(pair['success_rescues']), 1)
        self.assertEqual(len(pair['success_regressions']), 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failure_cases.png'
            plot_failure_rates(loaded, output)
            self.assertGreater(output.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
