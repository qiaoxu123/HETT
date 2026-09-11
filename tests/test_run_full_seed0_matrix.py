import unittest

from scripts.run_full_seed0_matrix import analysis_command, build_jobs


class FullSeedZeroMatrixTest(unittest.TestCase):
    def test_full_training_and_eval_jobs_are_predeclared(self):
        jobs = dict(build_jobs())
        self.assertEqual(set(jobs), {
            'recovery_off', 'recovery_on', 'grounding_disabled_control', 'grounding',
            'combined', 'hypothesis_disabled_control', 'hypothesis', 'bidirectional',
            'paper_loss_only', 'no_progress', 'neither_auxiliary',
        })
        for command in jobs.values():
            self.assertEqual(command[command.index('--epochs') + 1], '20')
            self.assertEqual(command[command.index('--max-episodes') + 1], '0')
            self.assertEqual(command[command.index('--save-every') + 1], '20')
            self.assertEqual(command[command.index('--seed') + 1], '0')
            self.assertNotIn('--include_test_unseen', command)
        for name in ('recovery_off', 'recovery_on', 'grounding_disabled_control',
                     'combined', 'hypothesis_disabled_control'):
            self.assertIn('--phase', jobs[name])
            self.assertIn('--checkpoint', jobs[name])
        for name in ('grounding', 'hypothesis', 'bidirectional', 'paper_loss_only',
                     'no_progress', 'neither_auxiliary'):
            self.assertNotIn('--phase', jobs[name])

    def test_comparisons_include_controls_and_each_claim(self):
        command = analysis_command('/tmp/full-seed-zero-analysis')
        pairs = [command[index + 1] for index, value in enumerate(command) if value == '--pair']
        self.assertEqual(len(pairs), 11)
        self.assertIn('recovery_off:recovery_on', pairs)
        self.assertIn('corrected:grounding_control', pairs)
        self.assertIn('grounding_control:grounding', pairs)
        self.assertIn('corrected:hypothesis_control', pairs)
        self.assertIn('hypothesis_control:hypothesis', pairs)
        self.assertIn('corrected:bidirectional', pairs)
        self.assertIn('corrected:paper_loss_only', pairs)
        self.assertIn('corrected:no_progress', pairs)
        self.assertIn('corrected:neither_auxiliary', pairs)


if __name__ == '__main__':
    unittest.main()
