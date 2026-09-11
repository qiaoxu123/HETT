import unittest

from scripts.audit_experiment_completion import (
    EVAL_VARIANTS, QUEUE_RESULTS, SEEDS, TRAIN_VARIANTS, run_path,
)


class CompletionAuditTest(unittest.TestCase):
    def test_manifest_covers_every_three_seed_claim(self):
        self.assertEqual(SEEDS, (0, 17, 42))
        self.assertEqual(set(TRAIN_VARIANTS), {
            'corrected', 'grounding', 'hypothesis', 'bidirectional',
            'paper_loss_only', 'no_progress', 'neither_auxiliary',
        })
        self.assertEqual(set(EVAL_VARIANTS), {'recovery_off', 'recovery_on', 'combined'})
        paths = {str(run_path(mapping, name, seed))
                 for mapping in (TRAIN_VARIANTS, EVAL_VARIANTS)
                 for name in mapping for seed in SEEDS}
        self.assertEqual(len(paths), 30)

    def test_queue_manifest_reaches_frozen_final_test(self):
        self.assertEqual(len(QUEUE_RESULTS), 10)
        self.assertIn('smoke', QUEUE_RESULTS)
        self.assertIn('multiseed', QUEUE_RESULTS)
        self.assertIn('final_test', QUEUE_RESULTS)


if __name__ == '__main__':
    unittest.main()
