import unittest

from scripts.report_live_baseline import loss_breakdown


class LiveBaselineReportTest(unittest.TestCase):
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
