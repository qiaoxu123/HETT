import unittest

from scripts.run_full_corrected_baseline import build_command


class FullCorrectedBaselineQueueTest(unittest.TestCase):
    def test_declares_full_paper_aligned_seed_zero_run(self):
        command = build_command()
        expected = {'--epochs': '20', '--max-episodes': '0', '--save-every': '20',
                    '--seed': '0'}
        for option, value in expected.items():
            self.assertIn(option, command)
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn('--variant-arg=--disable_task_interaction', command)
        self.assertIn('01-teacher-fix', ' '.join(command))

    def test_wrapper_does_not_pass_wait_flag_to_locking_supervisor(self):
        command = build_command()
        self.assertNotIn('--wait-for-unit', command)
        self.assertNotIn('--require-status', command)


if __name__ == '__main__':
    unittest.main()
