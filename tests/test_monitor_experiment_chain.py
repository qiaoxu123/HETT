import unittest

from scripts.monitor_experiment_chain import signature


class ChainMonitorTest(unittest.TestCase):
    def test_signature_ignores_telemetry_churn_but_tracks_phase_changes(self):
        value = {
            'units': {'unit': {'ActiveState': 'active'}},
            'run_statuses': {'unit': {'phase': 'training', 'time': 'first'}},
            'baseline_status': {'phase': 'training', 'gpu': '10'},
            'baseline_completed_epochs': 1,
            'alerts': [],
        }
        first = signature(value)
        value['baseline_status']['gpu'] = '90'
        value['run_statuses']['unit']['time'] = 'second'
        self.assertEqual(first, signature(value))
        value['baseline_completed_epochs'] = 2
        self.assertNotEqual(first, signature(value))

        second = signature(value)
        value['run_statuses']['unit']['phase'] = 'complete'
        self.assertNotEqual(second, signature(value))


if __name__ == '__main__':
    unittest.main()
