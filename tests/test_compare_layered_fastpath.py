import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_layered_fastpath import compare


class CompareLayeredFastpathTests(unittest.TestCase):
    def test_matched_manifest_and_step_count_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / 'before'
            new = Path(tmp) / 'after'
            old.mkdir()
            new.mkdir()
            for directory in (old,new):
                (directory/'manifest.json').write_text(json.dumps({
                    'initial_sha256':'one-checkpoint'
                }))
                for batch in (2,8):
                    (directory/f'gpu_batch{batch}.json').write_text(json.dumps({
                        'batch_size':batch,'optimizer_steps':2,
                        'seconds_per_batch':4.0 if directory==old else 2.0,
                        'peak_vram_allocated_bytes':2 * 1024**3,
                    }))
            results = compare(old,new)
            self.assertEqual([x['speedup'] for x in results],[2.0,2.0])
            (new/'manifest.json').write_text(json.dumps({
                'initial_sha256':'other-checkpoint'
            }))
            with self.assertRaises(ValueError):
                compare(old,new)


if __name__=='__main__':
    unittest.main()
