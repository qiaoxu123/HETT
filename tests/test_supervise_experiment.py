import argparse
from pathlib import Path
import unittest

from scripts.supervise_experiment import build_commands


class SupervisorCommandTest(unittest.TestCase):
    def test_variant_arguments_apply_to_train_and_eval(self):
        args = argparse.Namespace(seed=17, max_episodes=4, epochs=1, save_every=1,
                                  checkpoint=None,
                                  variant_arg=['--disable_task_interaction', '--enable_region_grounding'],
                                  train_variant_arg=[])
        train, evaluation = build_commands(Path('/snapshot'), Path('/run'), args)
        for command in (train, evaluation):
            self.assertIn('--seed', command)
            self.assertIn('17', command)
            self.assertIn('--enable_region_grounding', command)
            self.assertIn('--disable_task_interaction', command)
        self.assertIn('/run/checkpoints/best_val_unseen', evaluation)

    def test_eval_only_checkpoint_is_explicit(self):
        args = argparse.Namespace(seed=0, max_episodes=8, epochs=1, save_every=1,
                                  checkpoint='/parent/best', variant_arg=[], train_variant_arg=[])
        _, evaluation = build_commands(Path('/snapshot'), Path('/run'), args)
        self.assertIn('/parent/best', evaluation)

    def test_train_only_arguments_do_not_override_evaluation_checkpoint(self):
        args = argparse.Namespace(
            seed=0, max_episodes=8, epochs=8, save_every=1, checkpoint=None,
            variant_arg=['--coarse_to_fine_target'],
            train_variant_arg=['--checkpoint', '/parent/latest'],
        )
        train, evaluation = build_commands(Path('/snapshot'), Path('/run'), args)
        self.assertEqual(train[train.index('--checkpoint') + 1], '/parent/latest')
        self.assertEqual(
            evaluation[evaluation.index('--checkpoint') + 1],
            '/run/checkpoints/best_val_unseen',
        )
        self.assertEqual(evaluation.count('--checkpoint'), 1)


if __name__ == '__main__':
    unittest.main()
