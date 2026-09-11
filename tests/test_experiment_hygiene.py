import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from multiagent.parser import parse_args


ROOT = Path(__file__).resolve().parents[1]


class ExperimentHygieneTest(unittest.TestCase):
    def test_validation_passes_split_name_to_agent(self):
        source = (ROOT / 'multiagent/main.py').read_text()
        self.assertGreaterEqual(source.count('test(loader, env_name=env_name'), 2)

    def test_save_every_guards_immutable_epoch_archive(self):
        source = (ROOT / 'multiagent/main.py').read_text()
        tree = ast.parse(source)
        guarded_links = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and any(
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == 'link'
                    for child in ast.walk(node)):
                guarded_links.append(ast.unparse(node.test))
        self.assertTrue(any('save_every' in condition for condition in guarded_links))

    def test_test_split_does_not_store_region_ground_truth(self):
        source = (ROOT / 'multiagent/agent.py').read_text()
        self.assertIn("if 'test' not in self.env_name", source)
        self.assertIn("traj[i]['gt_region'].append", source)

    def test_instruction_override_is_explicit_and_keyed_by_episode(self):
        source = (ROOT / 'multiagent/agent.py').read_text()
        self.assertIn("instruction_overrides.get(ob['id'], ob['instruction'])", source)

    def test_recovery_cannot_silently_change_training_policy(self):
        with patch.object(sys, 'argv', ['test', '--mode', 'train', '--enable_stage_recovery']):
            with self.assertRaises(SystemExit):
                parse_args()
        with patch.object(sys, 'argv', ['test', '--mode', 'eval', '--enable_stage_recovery']):
            self.assertTrue(parse_args().enable_stage_recovery)


if __name__ == '__main__':
    unittest.main()
