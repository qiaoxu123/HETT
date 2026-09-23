import ast
from pathlib import Path


def test_stop_condition_is_not_an_elif_branch():
    tree = ast.parse(Path("multiagent/active_altitude/controller.py").read_text())
    assignments = [node for node in ast.walk(tree)
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "stop_evidence"
                           for target in node.targets)]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.BoolOp)
