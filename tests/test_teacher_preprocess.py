import numpy as np

from multiagent.space import Point2D, Point3D, Pose4D
from multiagent.teacher.preprocess import find_stable_arrival, optimize_teacher_path


SQUARE = [Point2D(40, -5), Point2D(50, -5), Point2D(50, 5), Point2D(40, 5)]


def points(values):
    return [Point3D(x, y, 50) for x, y in values]


def test_requires_stable_arrival():
    route = points([(0, 0), (35, 0), (0, 0), (35, 0), (36, 0)])
    assert find_stable_arrival(route, [SQUARE], radius=5, stable_points=2) == 3
    assert find_stable_arrival(route[:4], [SQUARE], radius=5, stable_points=2) is None


def test_splits_route_and_respects_phase_budgets():
    route = points([(x, 0) for x in range(0, 81, 5)])
    result = optimize_teacher_path(route, [SQUARE], arrival_radius=5,
                                   coarse_moves=4, local_moves=3)
    assert result is not None
    assert result.stage_boundary <= 4
    assert len(result.poses) - result.stage_boundary - 1 <= 3
    assert np.isclose(result.poses[0].x, 0)
    assert np.isclose(result.poses[-1].x, 80)


def test_removes_returning_loop_without_losing_endpoint():
    route = points([(0, 0), (20, 0), (35, 0), (45, 0), (55, 0),
                    (55, 20), (45, 20), (55, 20), (65, 10), (75, 0)])
    result = optimize_teacher_path(route, [SQUARE], arrival_radius=5,
                                   coarse_moves=10, local_moves=10)
    assert result is not None
    assert result.cleaned_points < result.source_points
    assert result.poses[-1].xy == Point2D(75, 0)


def test_no_landmark_arrival_preserves_original_by_returning_none():
    route = points([(0, 0), (10, 0), (20, 0)])
    assert optimize_teacher_path(route, [SQUARE], arrival_radius=5) is None


def test_retains_supplied_human_yaw_at_selected_nodes():
    route = [Pose4D(x, 0, 50, x / 100) for x in range(0, 81, 5)]
    result = optimize_teacher_path(route, [SQUARE], arrival_radius=5,
                                   coarse_moves=4, local_moves=3)
    assert result is not None
    assert all(np.isclose(pose.yaw, pose.x / 100) for pose in result.poses)
