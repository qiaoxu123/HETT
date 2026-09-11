import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from multiagent.parser import parse_args


ROOT = Path(__file__).resolve().parents[1]


class DevelopmentSplitPolicyTest(unittest.TestCase):
    def parse(self, *extra):
        with patch.object(sys, 'argv', ['split-policy-test', '--mode', 'eval', *extra]):
            return parse_args()

    def test_test_unseen_requires_explicit_opt_in(self):
        self.assertFalse(self.parse().include_test_unseen)
        self.assertTrue(self.parse('--include_test_unseen').include_test_unseen)

    def test_validation_builder_uses_opt_in_flag(self):
        source = (ROOT / 'multiagent/main.py').read_text()
        self.assertIn("val_env_names = ['val_seen', 'val_unseen']", source)
        self.assertIn('if args.include_test_unseen:', source)
        self.assertIn("val_env_names.append('test_unseen')", source)


if __name__ == '__main__':
    unittest.main()

