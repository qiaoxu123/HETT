import math

from independent_nav.controller import (
    GeometricGoalController,
    MOVE_FORWARD,
    STOP,
    TURN_LEFT,
)


def test_controller_reaches_goal_with_discrete_actions():
    result = GeometricGoalController().plan((0.0, 0.0), 0.0, (30.0, 20.0))
    assert result.actions[-1] == STOP
    assert math.dist(result.path_xy[-1], (30.0, 20.0)) <= 5.0
    assert MOVE_FORWARD in result.actions
    assert TURN_LEFT in result.actions


def test_controller_stops_immediately_inside_radius():
    result = GeometricGoalController().plan((0.0, 0.0), 1.0, (3.0, 0.0))
    assert result.actions == [STOP]
    assert result.path_xy == [(0.0, 0.0)]
