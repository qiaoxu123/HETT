import unittest
from pathlib import Path
import tempfile

from scripts.analyze_training_dynamics import aggregate, plot, weighted_record


ARGS = {'direction_loss_weight': 1.5, 'progress_loss_weight': .1,
        'goal_loss_weight': 2., 'target_loss_weight': .1}


def epoch(number, il_loss=4.6, sr=10):
    return weighted_record({
        'epoch': number, 'elapsed_seconds': number * 100, 'il_loss': il_loss,
        'direction_loss': 2, 'progress_loss': 1, 'goal_loss': .5,
        'target_loss': 5,
        'validation': {'val_unseen': {'sr': sr, 'spl': sr / 2, 'ne': 50 - sr}},
    }, ARGS)


class TrainingDynamicsTest(unittest.TestCase):
    def test_weighted_components_reconstruct_total(self):
        result = epoch(1)
        self.assertAlmostEqual(sum(result['weighted'].values()), result['il_loss'])
        self.assertAlmostEqual(sum(result['weighted_share_percent'].values()), 100)
        self.assertEqual(result['weighted']['progress'], .1)

    def test_aggregate_keeps_seed_curves_and_components(self):
        loaded = {'base': {
            0: {'epochs': [epoch(1, sr=10), epoch(2, sr=20)], 'summary': {'seed': 0}},
            17: {'epochs': [epoch(1, sr=20), epoch(2, sr=30)], 'summary': {'seed': 17}},
            42: {'epochs': [epoch(1, sr=30), epoch(2, sr=40)], 'summary': {'seed': 42}},
        }}
        result = aggregate(loaded)['base']
        self.assertEqual(result['curves']['val_unseen_sr']['mean'], [20, 30])
        self.assertEqual(result['curves']['val_unseen_sr']['std'], [10, 10])
        self.assertEqual(set(result['seeds']), {'0', '17', '42'})
        self.assertEqual(len(result['final_weighted_components']['target']), 3)
        with tempfile.TemporaryDirectory() as directory:
            plot({'base': result}, Path(directory))
            self.assertGreater((Path(directory) / 'training_curves.png').stat().st_size, 0)
            self.assertGreater((Path(directory) / 'weighted_loss_components.png').stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
