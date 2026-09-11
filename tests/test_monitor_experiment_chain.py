import unittest

from scripts.monitor_experiment_chain import batch_health, signature


class ChainMonitorTest(unittest.TestCase):
    def test_batch_health_distinguishes_spike_from_sustained_problem(self):
        rows = [
            {'epoch': 2, 'batch': 100, 'recent_il_loss': 8., 'grad_norm': 120.},
            {'epoch': 3, 'batch': 100, 'recent_il_loss': 9., 'grad_norm': 130.},
            {'epoch': 3, 'batch': 200, 'recent_il_loss': 8., 'grad_norm': 140.},
            {'epoch': 3, 'batch': 300, 'recent_il_loss': 7., 'grad_norm': 30.},
        ]
        health = batch_health(rows)
        self.assertEqual(health['epoch'], 3)
        self.assertEqual(health['samples'], 3)
        self.assertEqual(health['trailing_grad_over_100'], 0)
        rows[-1]['grad_norm'] = float('inf')
        self.assertEqual(batch_health(rows)['nonfinite_samples'], 1)

    def test_batch_health_detects_three_consecutive_large_gradients(self):
        rows = [{'epoch': 1, 'batch': batch, 'recent_il_loss': 8., 'grad_norm': grad}
                for batch, grad in ((100, 20.), (200, 101.), (300, 120.), (400, 150.))]
        self.assertEqual(batch_health(rows)['trailing_grad_over_100'], 3)

    def test_signature_ignores_telemetry_churn_but_tracks_phase_changes(self):
        value = {
            'units': {'unit': {'ActiveState': 'active'}},
            'run_statuses': {'unit': {'phase': 'training', 'time': 'first'}},
            'experiments': {'experiment': {'phase': 'training', 'time': 'first'}},
            'baseline_status': {'phase': 'training', 'gpu': '10'},
            'baseline_completed_epochs': 1,
            'alerts': [],
        }
        first = signature(value)
        value['baseline_status']['gpu'] = '90'
        value['run_statuses']['unit']['time'] = 'second'
        value['experiments']['experiment']['time'] = 'second'
        self.assertEqual(first, signature(value))
        value['baseline_completed_epochs'] = 2
        self.assertNotEqual(first, signature(value))

        second = signature(value)
        value['run_statuses']['unit']['phase'] = 'complete'
        self.assertNotEqual(second, signature(value))


if __name__ == '__main__':
    unittest.main()
