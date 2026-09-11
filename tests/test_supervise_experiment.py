import argparse
from pathlib import Path
import unittest

from scripts.supervise_experiment import build_commands


class SupervisorCommandTest(unittest.TestCase):
    def test_variant_arguments_apply_to_train_and_eval(self):
        args = argparse.Namespace(seed=17, max_episodes=4, epochs=1, save_every=1,
                                  checkpoint=None,
                                  variant_arg=['--disable_task_interaction', '--enable_region_grounding'])
        train, evaluation = build_commands(Path('/snapshot'), Path('/run'), args)
        for command in (train, evaluation):
            self.assertIn('--seed', command)
            self.assertIn('17', command)
            self.assertIn('--enable_region_grounding', command)
            self.assertIn('--disable_task_interaction', command)
        self.assertIn('/run/checkpoints/best_val_unseen', evaluation)

    def test_eval_only_checkpoint_is_explicit(self):
        args = argparse.Namespace(seed=0, max_episodes=8, epochs=1, save_every=1,
                                  checkpoint='/parent/best', variant_arg=[])
        _, evaluation = build_commands(Path('/snapshot'), Path('/run'), args)
        self.assertIn('/parent/best', evaluation)


if __name__ == '__main__':
    unittest.main()
