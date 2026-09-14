import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_validation_queue.py'
SPEC = importlib.util.spec_from_file_location('control_run_validation_queue', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
completed_run = MODULE.completed_run
run_directory = MODULE.run_directory
write_or_verify_manifest = MODULE.write_or_verify_manifest


class ValidationQueueResumeTest(unittest.TestCase):
    def command(self, run):
        return ['python', 'supervisor.py', '--run-dir', str(run)]

    def test_completed_run_is_safe_to_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            run.mkdir()
            (run / 'status.json').write_text('{"phase": "complete"}')
            command = self.command(run)
            self.assertEqual(run_directory(command), run)
            self.assertTrue(completed_run(command))
            (run / 'status.json').write_text('{"phase": "failed"}')
            self.assertFalse(completed_run(command))

    def test_restart_rejects_a_changed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'jobs.json'
            jobs = [('first', self.command(Path(directory) / 'run'))]
            write_or_verify_manifest(path, jobs)
            write_or_verify_manifest(path, jobs)
            with self.assertRaises(RuntimeError):
                write_or_verify_manifest(path, [('changed', jobs[0][1])])
            self.assertEqual(json.loads(path.read_text())[0]['name'], 'first')


if __name__ == '__main__':
    unittest.main()
