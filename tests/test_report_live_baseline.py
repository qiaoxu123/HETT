import unittest

from scripts.report_live_baseline import gradient_health, loss_breakdown, stage_diagnostics


class LiveBaselineReportTest(unittest.TestCase):
    def test_stage_diagnostics_separates_success_gain_from_distance_degradation(self):
        result = stage_diagnostics({'val_unseen': {
            'lengths': 100, 'stage2_length': 20, 'stage1_ne': 57.2, 'ne': 60.57,
            'sr1': 12.31, 'sr': 16.28, 'oracle_sr': 38.08,
        }})['val_unseen']
        self.assertAlmostEqual(result['fine_refinement_ne_change_m'], 3.37)
        self.assertAlmostEqual(result['fine_refinement_sr_change_pp'], 3.97)
        self.assertAlmostEqual(result['oracle_to_final_sr_gap_pp'], 21.8)
        self.assertEqual(result['stage2_path_share_percent'], 20)

    def test_gradient_health_records_long_tail_without_calling_it_divergence(self):
        result = gradient_health([
            {'epoch': 1, 'grad_norm': 20, 'recent_il_loss': 8},
            {'epoch': 1, 'grad_norm': 120, 'recent_il_loss': 9},
            {'epoch': 2, 'grad_norm': 30, 'recent_il_loss': 7},
        ])
        self.assertTrue(result['1']['all_finite'])
        self.assertEqual(result['1']['above_clip_40'], 1)
        self.assertEqual(result['1']['above_100'], 1)
        self.assertEqual(result['1']['grad_max'], 120)
        self.assertEqual(result['2']['above_clip_40'], 0)

    def test_recovers_hidden_target_loss_and_shares(self):
        row = {'il_loss': 10.0, 'direction_loss': 4.0, 'progress_loss': 2.0,
               'goal_loss': 1.0}
        result = loss_breakdown(row)
        self.assertAlmostEqual(result['raw']['target'], 18.0)
        self.assertAlmostEqual(sum(result['share_percent'].values()), 100.0)
        self.assertAlmostEqual(result['reconstructed_il_loss'], 10.0)

    def test_rejects_impossible_negative_remainder(self):
        row = {'il_loss': 1.0, 'direction_loss': 2.0, 'progress_loss': 0.0,
               'goal_loss': 0.0}
        with self.assertRaises(ValueError):
            loss_breakdown(row)


if __name__ == '__main__':
    unittest.main()
