import unittest
from pathlib import Path
import tempfile

from scripts.run_multiseed_confirmation import (
    MIN_FREE_GIB, ROOT, SEEDS, STORAGE_ROOT, analysis_command, build_jobs,
    externalize_run, prepare_queue_dir, publish_logical_run,
    training_analysis_command,
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

    def test_additional_seed_runs_are_externalized_without_changing_protocol(self):
        for unused_name, command in build_jobs():
            actual, logical, physical = externalize_run(command)
            self.assertEqual(logical, Path(command[command.index('--run-dir') + 1]))
            self.assertTrue(logical.is_relative_to(ROOT))
            self.assertTrue(physical.is_relative_to(STORAGE_ROOT))
            self.assertEqual(actual[actual.index('--epochs') + 1], '20')
            self.assertEqual(actual[actual.index('--max-episodes') + 1], '0')
            self.assertEqual(command[command.index('--run-dir') + 1], str(logical))

    def test_completed_physical_run_is_published_as_a_logical_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = root / 'storage/run'
            physical.mkdir(parents=True)
            (physical / 'status.json').write_text('{}')
            logical = root / 'worktree/runs/run'
            publish_logical_run(logical, physical)
            self.assertTrue(logical.is_symlink())
            self.assertEqual(logical.resolve(), physical.resolve())
            self.assertTrue((logical / 'status.json').is_file())

    def test_queue_restart_is_allowed_only_before_any_job_started(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / 'queue'
            queue.mkdir()
            (queue / 'status.json').write_text('{"phase": "waiting"}')
            prepare_queue_dir(queue)
            (queue / 'events.jsonl').write_text('started\n')
            with self.assertRaises(FileExistsError):
                prepare_queue_dir(queue)


if __name__ == '__main__':
    unittest.main()
