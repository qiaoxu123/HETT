import math

import pytest

from multiagent.cityreferobject import CityReferObject, ProcessedDescription
from multiagent.dataset.episode import (
    Episode,
    ReverseTeacherEpisode,
    metric_direction_name,
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
