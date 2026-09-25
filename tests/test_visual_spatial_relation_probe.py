import numpy as np

from scripts.train_visual_spatial_relation_probe import (
    candidate_boundary_directions,
    landmark_mask,
    same_map_shift,
    symmetric_candidates,
)


def test_symmetric_candidates_mirror_around_landmark():
    result = symmetric_candidates(np.array([0.5, 0.5]), np.array([0.2, -0.1]))

    assert np.allclose(result, [[0.7, 0.4], [0.3, 0.4],
                                [0.7, 0.6], [0.3, 0.6]])


def test_wrong_mask_control_stays_within_map():
    shifted = same_map_shift(["a", "b", "a", "b"])

    assert shifted.tolist() == [2, 3, 0, 1]


def test_candidate_direction_uses_nearest_landmark_boundary(monkeypatch):
    class Bounds:
        x_min = 0.0
        y_max = 410.0

    module = __import__("scripts.audit_spatial_reference_frames", fromlist=["MAP_BOUNDS"])
    monkeypatch.setitem(module.MAP_BOUNDS, "map", Bounds())
    contour = [[164, 246], [246, 246], [246, 164], [164, 164]]

    result = candidate_boundary_directions(
        np.array([[0.8, 0.5]], dtype=np.float32), [contour], "map")

    assert result[0, 0] > 0.99
    assert abs(result[0, 1]) < 0.01
