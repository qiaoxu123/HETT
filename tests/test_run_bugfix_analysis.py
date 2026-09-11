import unittest

from scripts.run_bugfix_analysis import build_commands


class BugfixAnalysisTest(unittest.TestCase):
    def test_analysis_is_paired_and_kept_separate_from_innovations(self):
        commands = build_commands('/tmp/bugfix-analysis')
        self.assertEqual(set(commands), {'training', 'navigation'})
        self.assertIn('original_buggy:corrected', commands['navigation'])
        joined = ' '.join(sum(commands.values(), []))
        self.assertIn('original_buggy', joined)
        self.assertIn('teacher_fix_full_s0', joined)
        self.assertNotIn('grounding_full', joined)
        self.assertNotIn('bidir_full', joined)
        self.assertNotIn('test_unseen', joined)


if __name__ == '__main__':
    unittest.main()
