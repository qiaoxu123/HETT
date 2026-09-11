import unittest

from multiagent.space import Point2D, Pose4D
from scripts.analyze_validation_matrix import episode_rows, paired_summary, summarize_rows


def pose(x):
    return Pose4D(x, 0, 50, 0)


class ValidationAnalysisTest(unittest.TestCase):
    def test_records_hit_then_lost_and_auxiliary_diagnostics(self):
        predictions = {('map_a', 1, 1): {
            'goal': Point2D(0, 0),
            'trajectory': [pose(30), pose(10), pose(25)],
            'gt_trajectory': [pose(30), pose(0)],
            'control_events': ['fine', 'coarse', 'coarse'],
            'region_prediction': [2, 49],
            'gt_region': [2, 49],
            'hypothesis_indices': [[1, 2], [2, 1]],
            'hypothesis_confidence': [.2, .8],
        }}
        rows = episode_rows(predictions)
        row = next(iter(rows.values()))
        self.assertTrue(row['hit_then_lost'])
        self.assertEqual(row['recoveries'], 1)
        summary = summarize_rows(rows)
        self.assertEqual(summary['region_accuracy_percent'], 100)
        self.assertEqual(summary['hypothesis_map_change_percent'], 100)
        self.assertAlmostEqual(summary['mean_hypothesis_confidence'], .5)

    def test_paired_result_counts_wins_and_losses(self):
        left = {'a': {'success': False, 'final_distance': 30., 'map': 'm1'},
                'b': {'success': True, 'final_distance': 10., 'map': 'm2'}}
        right = {'a': {'success': True, 'final_distance': 5., 'map': 'm1'},
                 'b': {'success': False, 'final_distance': 25., 'map': 'm2'}}
        result = paired_summary(left, right, bootstrap_samples=100)
        self.assertEqual(result['success_wins'], 1)
        self.assertEqual(result['success_losses'], 1)
        self.assertEqual(result['success_delta_pp'], 0)
        self.assertEqual(result['mean_final_distance_delta_m'], -5)


if __name__ == '__main__':
    unittest.main()
