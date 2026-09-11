import json
import unittest
from pathlib import Path
import tempfile

from scripts.audit_experiment_completion import (
    EVAL_VARIANTS, EXPECTED_HEADS, QUEUE_RESULTS, SEEDS, TRAIN_VARIANTS,
    command_arguments, digest, run_path, validate_checkpoint_hashes,
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

    def test_checkpoint_hash_audit_detects_later_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / 'checkpoints'
            checkpoint.mkdir()
            rows = []
            for name, content in [('epoch_20.pt', b'epoch'),
                                  ('best_val_unseen', b'best')]:
                path = checkpoint / name
                path.write_bytes(content)
                rows.append({'file': name, 'bytes': len(content), 'sha256': digest(path)})
            (run / 'checkpoint_hashes.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows))
            self.assertTrue(validate_checkpoint_hashes(run)[0])
            (checkpoint / 'best_val_unseen').write_bytes(b'overwritten')
            passed, evidence = validate_checkpoint_hashes(run)
            self.assertFalse(passed)
            self.assertFalse(evidence['files']['best_val_unseen']['sha256_match'])

    def test_command_argument_audit_checks_nested_commands_not_json_substrings(self):
        commands = {'train': ['bash', 'train.sh'],
                    'evaluation': ['bash', 'eval.sh', '--include_test_unseen'],
                    'environment_overrides': {'NOTE': 'not an argument'}}
        self.assertIn('--include_test_unseen', command_arguments(commands))
        self.assertNotIn('--include_test', command_arguments(commands))


if __name__ == '__main__':
    unittest.main()
