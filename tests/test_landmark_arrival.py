import math

from multiagent.maps.landmark_map import distance_to_landmark_contours
from multiagent.space import Point2D


SQUARE = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]


def test_inside_and_boundary_are_zero_distance():
    assert distance_to_landmark_contours(Point2D(5, 5), [SQUARE]) == 0
    assert distance_to_landmark_contours(Point2D(10, 5), [SQUARE]) == 0


def test_outside_distance_is_in_map_metres():
    assert distance_to_landmark_contours(Point2D(13, 5), [SQUARE]) == 3
    assert math.isclose(distance_to_landmark_contours(Point2D(13, 14), [SQUARE]), 5)


def test_uses_nearest_of_multiple_landmarks():
    other = [Point2D(20, 0), Point2D(30, 0), Point2D(30, 10), Point2D(20, 10)]
    assert distance_to_landmark_contours(Point2D(18, 5), [SQUARE, other]) == 2


def test_no_landmark_cannot_confirm_arrival():
    assert math.isinf(distance_to_landmark_contours(Point2D(0, 0), []))
