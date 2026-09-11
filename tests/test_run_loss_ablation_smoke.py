import unittest

from scripts.run_loss_ablation_smoke import build_jobs


class LossAblationQueueTest(unittest.TestCase):
    def test_four_isolated_weight_protocols(self):
        jobs = {name: command for name, command in build_jobs()}
        self.assertEqual(set(jobs), {'released', 'paper_loss_only', 'no_progress', 'neither'})
        rendered = {name: ' '.join(command) for name, command in jobs.items()}
        self.assertNotIn('--target_loss_weight', rendered['released'])
        self.assertNotIn('--progress_loss_weight', rendered['released'])
        self.assertIn('--target_loss_weight --variant-arg=0', rendered['paper_loss_only'])
        self.assertIn('--progress_loss_weight --variant-arg=0', rendered['no_progress'])
        self.assertIn('--target_loss_weight --variant-arg=0', rendered['neither'])
        self.assertIn('--progress_loss_weight --variant-arg=0', rendered['neither'])

    def test_all_jobs_are_short_and_disable_interaction(self):
        for _, command in build_jobs():
            self.assertIn('--max-episodes', command)
            self.assertEqual(command[command.index('--max-episodes') + 1], '4')
            self.assertIn('--epochs', command)
            self.assertEqual(command[command.index('--epochs') + 1], '1')
            self.assertIn('--variant-arg=--disable_task_interaction', command)


if __name__ == '__main__':
    unittest.main()
