import unittest

from scripts.run_post_smoke_ablations import build_jobs


class PostSmokeAblationTest(unittest.TestCase):
    def test_declares_all_grounding_sensitivity_checks(self):
        jobs = dict(build_jobs())
        self.assertEqual(set(jobs), {
            'grounding_off', 'grounding_shuffle_language', 'grounding_shuffle_visual'})
        self.assertNotIn('--enable_region_grounding', jobs['grounding_off'])
        self.assertIn('--variant-arg=--enable_region_grounding', jobs['grounding_shuffle_language'])
        self.assertIn('--variant-arg=shuffle_language', jobs['grounding_shuffle_language'])
        self.assertIn('--variant-arg=shuffle_visual', jobs['grounding_shuffle_visual'])


if __name__ == '__main__':
    unittest.main()
