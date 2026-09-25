import ast
from pathlib import Path


def test_low_altitude_history_has_an_explicit_global_memory_mask():
    agent_source = Path("multiagent/agent.py").read_text()
    model_source = Path("multiagent/models/ET_haa.py").read_text()
    assert "grid_valid" in agent_source
    assert "grid_valid" in model_source
    assert "cell_mask = (grid_map_indexs[b] == i) & grid_valid[b].bool()" in model_source
    assert "frozen_coarse_goals" in agent_source


def test_fine_altitude_does_not_replace_existing_stop_condition():
    source = Path("multiagent/agent.py").read_text()
    tree = ast.parse(source)
    stop_conditions = [node for node in ast.walk(tree)
                       if isinstance(node, ast.BoolOp)]
    assert "pred_progress_t[i] > self.args.progress_stop_threshold" in source
    assert any(
        "progress_stop_threshold" in ast.unparse(node)
        and "stage1_ended" in ast.unparse(node)
        for node in stop_conditions)


def test_fine_altitude_is_gated_by_the_existing_fine_navigation_stage():
    source = Path("multiagent/agent.py").read_text()
    assert "fine_altitude_enabled and stage1_ended[i] and not ended[i]" in source


def test_fine_altitude_development_path_keeps_test_unseen_sealed():
    source = Path("multiagent/main.py").read_text()
    assert "getattr(args, 'fine_altitude_refinement', False)" in source
    assert "getattr(args, 'eval_splits', None)" in source
    assert "['val_seen', 'val_unseen']" in source

    parser_source = Path("multiagent/parser.py").read_text()
    assert "'test_unseen' in args.eval_splits" in parser_source
