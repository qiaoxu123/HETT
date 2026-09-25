import numpy as np

from independent_nav.projection import (
    image_to_world,
    transform_points,
    view_corners,
    world_to_image_homography,
)


def test_world_image_projection_round_trip():
    corners = view_corners(100.0, 200.0, 0.7, 50.0)
    transform = world_to_image_homography(corners, 128)
    points = np.asarray([[100.0, 200.0], [110.0, 190.0]], dtype=np.float32)
    restored = image_to_world(transform_points(points, transform), transform)
    assert np.allclose(restored, points, atol=1e-4)


def test_camera_center_maps_to_image_center():
    corners = view_corners(100.0, 200.0, 0.7, 50.0)
    transform = world_to_image_homography(corners, 128)
    image = transform_points([[100.0, 200.0]], transform)[0]
    assert np.allclose(image, [63.5, 63.5], atol=1e-4)
