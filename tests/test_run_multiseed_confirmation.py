import unittest

from scripts.run_multiseed_confirmation import MIN_FREE_GIB, SEEDS, build_jobs


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


if __name__ == '__main__':
    unittest.main()
