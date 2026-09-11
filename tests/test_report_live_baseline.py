import unittest

from scripts.report_live_baseline import gradient_health, loss_breakdown


class LiveBaselineReportTest(unittest.TestCase):
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
