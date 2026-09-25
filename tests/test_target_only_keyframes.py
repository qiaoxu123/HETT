import numpy as np

from multiagent.target_only import (
    resample_contour,
    select_keyframe_pose,
    target_prediction_metrics,
)


class XY:
    def __init__(self, x, y):
        self.x, self.y = x, y

    @property
    def xy(self):
        return self

    def dist_to(self, other):
        return float(np.hypot(self.x - other.x, self.y - other.y))


def test_near_ring_keyframe_stays_outside_success_circle():
    trajectory = [XY(distance, 0) for distance in (100, 55, 35, 15, 5)]
    selected = select_keyframe_pose(trajectory, XY(0, 0), "near_ring")
    assert 25 <= selected.dist_to(XY(0, 0)) <= 60


def test_near_ring_fallback_does_not_choose_inside_pose_when_possible():
    trajectory = [XY(distance, 0) for distance in (90, 70, 10, 2)]
    selected = select_keyframe_pose(trajectory, XY(0, 0), "near_ring")
    assert selected.dist_to(XY(0, 0)) == 70


def test_final_keyframe_is_exact_human_trajectory_endpoint():
    trajectory = [XY(100, 0), XY(50, 0), XY(7, 0)]
    assert select_keyframe_pose(trajectory, XY(0, 0), "final") is trajectory[-1]


def test_resample_contour_has_fixed_shape_and_boundary_points():
    square = [XY(0, 0), XY(1, 0), XY(1, 1), XY(0, 1)]
    result = resample_contour(square, 16)
    assert result.shape == (16, 2)
    assert np.all((result >= 0) & (result <= 1))
    assert np.all(np.isclose(result[:, 0], 0) | np.isclose(result[:, 0], 1)
                  | np.isclose(result[:, 1], 0) | np.isclose(result[:, 1], 1))


def test_target_metrics_report_20m_accuracy():
    metrics = target_prediction_metrics([0, 10, 25, 100])
    assert metrics["count"] == 4
    assert metrics["hit5"] == 25.0
    assert metrics["hit20"] == 50.0
    assert metrics["median_error_m"] == 17.5
