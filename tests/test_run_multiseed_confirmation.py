import unittest

from scripts.run_multiseed_confirmation import (
    MIN_FREE_GIB, SEEDS, analysis_command, build_jobs, training_analysis_command,
)


class MultiSeedConfirmationTest(unittest.TestCase):
    def test_every_claim_has_two_additional_full_seeds(self):
        jobs = build_jobs()
        self.assertEqual(SEEDS, (17, 42))
        self.assertEqual(len(jobs), 20)
        names = [name for name, unused in jobs]
        for seed in SEEDS:
            for prefix in ('teacher_fix', 'recovery_off', 'recovery_on', 'grounding',
                           'combined', 'hypothesis', 'bidirectional', 'paper_loss_only',
                           'no_progress', 'neither_auxiliary'):
                self.assertIn(f'{prefix}_s{seed}', names)

    def test_protocol_and_test_split_are_locked(self):
        for name, command in build_jobs():
            seed = name.rsplit('_s', 1)[1]
            self.assertEqual(command[command.index('--seed') + 1], seed)
            self.assertEqual(command[command.index('--epochs') + 1], '20')
            self.assertEqual(command[command.index('--max-episodes') + 1], '0')
            self.assertEqual(command[command.index('--save-every') + 1], '20')
            self.assertNotIn('--include_test_unseen', command)
            if name.startswith(('recovery_', 'combined_')):
                self.assertIn('--phase', command)
                self.assertIn('--checkpoint', command)
        self.assertGreaterEqual(MIN_FREE_GIB, 12)

    def test_analysis_uses_all_three_seeds_without_test(self):
        command = analysis_command('/tmp/multiseed-analysis')
        runs = [command[index + 1] for index, value in enumerate(command) if value == '--run']
        pairs = [command[index + 1] for index, value in enumerate(command) if value == '--pair']
        self.assertEqual(len(runs), 30)
        self.assertEqual(len(pairs), 9)
        for seed in (0, 17, 42):
            self.assertIn(f'corrected:{seed}=', ' '.join(runs))
            self.assertIn(f'bidirectional:{seed}=', ' '.join(runs))
        self.assertNotIn('test_unseen', ' '.join(command))

        training = training_analysis_command('/tmp/training-analysis')
        training_runs = [training[index + 1] for index, value in enumerate(training)
                         if value == '--run']
        self.assertEqual(len(training_runs), 21)
        self.assertNotIn('recovery', ' '.join(training_runs))
        self.assertNotIn('combined', ' '.join(training_runs))


if __name__ == '__main__':
    unittest.main()
