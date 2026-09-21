from multiagent.stage_control import advance_teacher_stage1
from pathlib import Path


def test_teacher_cursors_are_independent_per_episode():
    steps = [0, 0]
    assert advance_teacher_stage1(steps, 0, 10, 100) == 10
    assert advance_teacher_stage1(steps, 0, 10, 100) == 20
    assert advance_teacher_stage1(steps, 1, 10, 100) == 10
    assert steps == [2, 1]


def test_cursor_falls_back_to_final_pose_when_route_is_exhausted():
    steps = [0]
    assert advance_teacher_stage1(steps, 0, 10, 5) == -1


def test_landmark_gate_does_not_short_circuit_progress_stop():
    source = (Path(__file__).resolve().parents[1] / 'multiagent/agent.py').read_text()
    predicate = "pred_progress_t[i] > 0.95 and self.feedback == 'student' and stage1_ended[i]"
    assert f"if {predicate}:" in source
    assert f"elif {predicate}:" not in source
