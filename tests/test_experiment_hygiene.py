import ast
from pathlib import Path
import unittest


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


if __name__ == '__main__':
    unittest.main()
