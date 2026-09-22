import math

from multiagent.stage_control import advance_teacher_stage1, target_progress
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
    assert "pred_progress_t[i] >= self.args.progress_stop_threshold" in source
    assert "elif (pred_progress_t[i] >= self.args.progress_stop_threshold" not in source


def test_progress_supervision_matches_distance_definition():
    cases = [(100, 0.0), (50, 0.5), (20, 0.8), (10, 0.9), (5, 0.95), (0, 1.0)]
    for distance_m, expected in cases:
        assert math.isclose(target_progress(distance_m), expected)


def test_progress_supervision_is_clipped():
    assert target_progress(150) == 0.0
    assert target_progress(-10) == 1.0


def test_success_radius_maps_to_aligned_stop_threshold():
    success_radius_m = 20
    assert math.isclose(target_progress(success_radius_m), 0.8)
