import unittest

from scripts.monitor_experiment_chain import signature


class ChainMonitorTest(unittest.TestCase):
    def test_signature_ignores_telemetry_churn_but_tracks_phase_changes(self):
        value = {
            'units': {'unit': {'ActiveState': 'active'}},
            'baseline_status': {'phase': 'training', 'gpu': '10'},
            'baseline_completed_epochs': 1,
            'alerts': [],
        }
        first = signature(value)
        value['baseline_status']['gpu'] = '90'
        self.assertEqual(first, signature(value))
        value['baseline_completed_epochs'] = 2
        self.assertNotEqual(first, signature(value))


if __name__ == '__main__':
    unittest.main()
