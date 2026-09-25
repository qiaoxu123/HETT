from pathlib import Path


def test_progress_stop_threshold_is_configurable_and_defaults_to_release_value():
    parser_source = Path("multiagent/parser.py").read_text()
    agent_source = Path("multiagent/agent.py").read_text()
    assert "--progress_stop_threshold" in parser_source
    assert "default=0.95" in parser_source
    assert "pred_progress_t[i] > self.args.progress_stop_threshold" in agent_source
    assert "pred_progress_t[i] > 0.95" not in agent_source

