import unittest

from scripts.run_hypothesis_ablations import build_jobs


class HypothesisAblationQueueTest(unittest.TestCase):
    def test_compares_module_and_temporal_memory(self):
        jobs = dict(build_jobs())
        self.assertEqual(set(jobs), {'hypothesis_off', 'hypothesis_no_temporal'})
        self.assertNotIn('--variant-arg=--enable_multi_hypothesis', jobs['hypothesis_off'])
        self.assertIn('--variant-arg=--enable_multi_hypothesis', jobs['hypothesis_no_temporal'])
        self.assertIn('--variant-arg=0', jobs['hypothesis_no_temporal'])


if __name__ == '__main__':
    unittest.main()
