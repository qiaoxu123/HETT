import unittest
from pathlib import Path
import tempfile

from scripts.audit_experiment_completion import (
    EVAL_VARIANTS, EXPECTED_HEADS, QUEUE_RESULTS, SEEDS, TRAIN_VARIANTS, digest, run_path,
)


class CompletionAuditTest(unittest.TestCase):
    def test_digest_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence'
            path.write_bytes(b'baseline evaluation hygiene')
            self.assertEqual(digest(path),
                             '568ebf9ecad1fec4e13914ce7e9403348c5d730dd333574884400f1f60d46cc9')

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
        self.assertEqual({worktree for worktree, unused in
                          (*TRAIN_VARIANTS.values(), *EVAL_VARIANTS.values())},
                         set(EXPECTED_HEADS))
        self.assertTrue(all(len(head) == 40 for head in EXPECTED_HEADS.values()))

    def test_queue_manifest_reaches_frozen_final_test(self):
        self.assertEqual(len(QUEUE_RESULTS), 11)
        self.assertIn('smoke', QUEUE_RESULTS)
        self.assertIn('multiseed', QUEUE_RESULTS)
        self.assertIn('final_test', QUEUE_RESULTS)
        self.assertIn('bugfix_analysis', QUEUE_RESULTS)


if __name__ == '__main__':
    unittest.main()
