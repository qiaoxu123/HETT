import math

import pytest
from types import SimpleNamespace

from multiagent.navigation_control import should_replan_stage2, should_stop_navigation
from multiagent.cityreferobject import CityReferObject, ProcessedDescription
from multiagent.dataset.episode import (
    Episode,
    ReverseTeacherEpisode,
    metric_direction_name,
    reverse_waypoint_direction,
    reverse_flight_trajectory,
)
from multiagent.space import Point2D, Point3D, Pose4D


def make_episode():
    target = CityReferObject(
        map_name='test_map',
        id=7,
        name='clock tower',
        object_type='building',
        position=Point3D(20.0, 10.0, 0.0),
        dimension=Point3D(2.0, 2.0, 4.0),
        descriptions=['The clock tower beside the library.'],
        contour=[Point2D(19, 9), Point2D(21, 9), Point2D(21, 11), Point2D(19, 11)],
        processed_descriptions=[ProcessedDescription(
            target='clock tower',
            landmarks=['library'],
            surroundings=['road'],
        )],
    )
    trajectory = [
        Pose4D(0.0, 0.0, 50.0, 0.1),
        Pose4D(10.0, 0.0, 50.0, 0.2),
        Pose4D(10.0, 10.0, 50.0, 0.3),
    ]
    return Episode(target, 0, trajectory, [0, 0, 0])


def test_reverse_positions_and_yaws_follow_reverse_motion():
    trajectory = make_episode().teacher_trajectory
    reversed_trajectory = reverse_flight_trajectory(trajectory)

    assert [(p.x, p.y) for p in reversed_trajectory] == [
        (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)
    ]
    assert reversed_trajectory[0].yaw == pytest.approx(-math.pi / 2)
    assert abs(abs(reversed_trajectory[1].yaw) - math.pi) < 1e-6
    assert reversed_trajectory[2].yaw == pytest.approx(reversed_trajectory[1].yaw)


def test_reverse_waypoint_direction_uses_local_motion_not_global_goal():
    trajectory = reverse_flight_trajectory(make_episode().teacher_trajectory)

    assert reverse_waypoint_direction(trajectory, step=0, stride=1) == pytest.approx(0.0)
    assert reverse_waypoint_direction(trajectory, step=1, stride=1) == pytest.approx(0.0)


def test_forward_and_reverse_progress_share_initial_distance_definition():
    forward = make_episode()
    reverse = ReverseTeacherEpisode(forward)

    assert forward.initial_distance_to_target > 0
    assert reverse.initial_distance_to_target > 0


def test_stage2_replans_when_belief_goal_moves_far_from_anchor():
    anchor = Point2D(0.0, 0.0)

    assert not should_replan_stage2(True, anchor, Point2D(10.0, 0.0), 15.0)
    assert should_replan_stage2(True, anchor, Point2D(20.0, 0.0), 15.0)
    assert not should_replan_stage2(False, anchor, Point2D(20.0, 0.0), 15.0)


def test_stop_requires_progress_distance_and_stable_goal():
    args = SimpleNamespace(
        progress_stop_threshold=0.95,
        stop_goal_distance_m=10.0,
        goal_stability_steps=2,
    )

    assert should_stop_navigation(True, 0.99, 5.0, 2, args)
    assert not should_stop_navigation(True, 0.90, 5.0, 2, args)
    assert not should_stop_navigation(True, 0.99, 15.0, 2, args)
    assert not should_stop_navigation(True, 0.99, 5.0, 1, args)


def test_reverse_metric_task_is_separate_from_landmark_semantics():
    reverse = ReverseTeacherEpisode(make_episode())

    assert reverse.target_position == Point3D(0.0, 0.0, 50.0)
    assert reverse.visual_alignment_text == 'clock tower'
    assert reverse.description_landmarks == ['library']
    assert 'clock tower' not in reverse.target_description.lower()
    assert 'starting point' in reverse.target_description.lower()


@pytest.mark.parametrize(
    ('delta', 'name'),
    [((1, 0), 'east'), ((0, 1), 'north'), ((-1, 0), 'west'), ((0, -1), 'south')],
)
def test_metric_direction_names(delta, name):
    assert metric_direction_name(*delta) == name
