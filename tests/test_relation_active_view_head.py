import numpy as np

from scripts.train_relation_active_view_head import local_coordinates


def test_local_coordinates_are_view_relative():
    points = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    value = local_coordinates(points, np.asarray([0.0, 0.0]),
                              np.asarray([1.0, 0.0]))
    assert np.allclose(value[0], [0.01, 0.0])
    assert np.allclose(value[1], [0.0, 0.01])
