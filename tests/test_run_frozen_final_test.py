import unittest

from scripts.run_frozen_final_test import (
    evaluation_command, final_analysis_command, freeze_candidate,
)


def summary(values, include_test=False):
    runs = {}
    for name, (sr, spl) in values.items():
        runs[name] = {'val_unseen': {'sr': {'mean': sr}, 'spl': {'mean': spl}}}
        if include_test:
            runs[name]['test_unseen'] = {'sr': {'mean': 99}}
    return {'runs': runs}


class FrozenFinalTest(unittest.TestCase):
    def base_values(self):
        return {
            'corrected': (20, 15), 'recovery_on': (22, 16), 'grounding': (24, 14.5),
            'combined': (25, 13), 'hypothesis': (23, 14), 'bidirectional': (26, 15),
            'paper_loss_only': (18, 14), 'no_progress': (21, 15),
            'neither_auxiliary': (17, 13),
        }

    def test_freezes_highest_eligible_validation_candidate(self):
        frozen = freeze_candidate(summary(self.base_values()))
        self.assertEqual(frozen['selected'], 'bidirectional')
        self.assertTrue(frozen['considered']['grounding']['eligible'])
        self.assertFalse(frozen['considered']['combined']['eligible'])
        self.assertFalse(frozen['test_metrics_observed'])

    def test_falls_back_to_baseline_and_rejects_prefreeze_test_metrics(self):
        values = self.base_values()
        values['grounding'] = (22.9, 16)
        values['hypothesis'] = (22.9, 16)
        values['bidirectional'] = (22.9, 16)
        self.assertEqual(freeze_candidate(summary(values))['selected'], 'corrected')
        with self.assertRaises(ValueError):
            freeze_candidate(summary(values, include_test=True))

    def test_test_split_is_explicit_only_in_frozen_evaluation(self):
        for name in self.base_values():
            command = evaluation_command(name, 17)
            self.assertIn('--phase', command)
            self.assertIn('--checkpoint', command)
            self.assertIn('--variant-arg=--include_test_unseen', command)
            self.assertEqual(command[command.index('--seed') + 1], '17')
        analysis = final_analysis_command('grounding', '/tmp/final-analysis')
        self.assertEqual(sum(value == '--run' for value in analysis), 6)
        self.assertIn('corrected:grounding', analysis)


if __name__ == '__main__':
    unittest.main()
