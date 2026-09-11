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
            'stage1_trajectory': [pose(30)],
            'stage2_trajectory': [pose(30), pose(10)],
            'gt_trajectory': [pose(30), pose(0)],
            'control_events': ['fine', 'coarse', 'coarse'],
            'progress': [.2, .4, .5],
            'region_prediction': [2, 49],
            'gt_region': [2, 49],
            'hypothesis_indices': [[1, 2], [2, 1]],
            'hypothesis_confidence': [.2, .8],
        }}
        rows = episode_rows(predictions)
        row = next(iter(rows.values()))
        self.assertTrue(row['hit_then_lost'])
        self.assertEqual(row['switches'], 1)
        self.assertEqual(row['recoveries'], 1)
        self.assertEqual(row['switch_distance_sum'], 30)
        self.assertTrue(row['stopped'])
        self.assertTrue(row['false_stop'])
        summary = summarize_rows(rows)
        self.assertEqual(summary['mean_switch_distance_m'], 30)
        self.assertEqual(summary['false_stop_percent_of_stops'], 100)
        self.assertEqual(summary['mean_predicted_progress_at_stop'], .5)
        self.assertEqual(summary['region_accuracy_percent'], 100)
        self.assertEqual(summary['hypothesis_map_change_percent'], 100)
        self.assertAlmostEqual(summary['mean_hypothesis_confidence'], .5)

    def test_original_one_way_stage_metrics_are_derived_without_new_trace_fields(self):
        predictions = {('map_a', 1, 2): {
            'goal': Point2D(0, 0),
            'trajectory': [pose(30), pose(20), pose(10)],
            'stage1_trajectory': [pose(30), pose(20)],
            'stage2_trajectory': [pose(20), pose(10)],
            'gt_trajectory': [pose(30), pose(0)],
            'progress': [.2, .7],
        }}
        row = next(iter(episode_rows(predictions).values()))
        self.assertEqual(row['switches'], 1)
        self.assertEqual(row['switch_distance_sum'], 20)
        self.assertFalse(row['stopped'])
        self.assertFalse(row['false_stop'])

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
