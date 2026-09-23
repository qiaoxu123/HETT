import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_altitude_view_benchmark.py"
SPEC = importlib.util.spec_from_file_location("altitude_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_view_area_scales_quadratically_with_altitude():
    small = MODULE.valid_polygon(MODULE.view_corners(10, 20, 0.3, 30))
    large = MODULE.valid_polygon(MODULE.view_corners(10, 20, 0.3, 60))
    assert np.isclose(large.area / small.area, 4.0)


def test_world_center_projects_to_image_center():
    corners = MODULE.view_corners(10, 20, 0.3, 60)
    transform = MODULE.world_to_image_homography(corners, 512)
    center = np.asarray([[[10, 20]]], dtype=np.float32)
    projected = MODULE.cv2.perspectiveTransform(center, transform)[0, 0]
    assert np.allclose(projected, [255.5, 255.5], atol=1e-3)


def test_digital_zoom_preserves_output_shape():
    image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    zoomed = MODULE.digital_zoom(image, low_altitude=30, high_altitude=120)
    assert zoomed.shape == image.shape


def test_out_of_view_polygon_produces_empty_mask():
    corners = MODULE.view_corners(0, 0, 0, 30)
    view = MODULE.valid_polygon(corners)
    target = MODULE.valid_polygon([[100, 100], [110, 100], [110, 110], [100, 110]])
    transform = MODULE.world_to_image_homography(corners, 224)
    mask = MODULE.rasterize_geometry(target.intersection(view), transform, 224)
    assert mask.sum() == 0


def test_select_distance_poses_deduplicates_short_trajectory():
    row = {
        "target_positions": [[0, 0, 0]],
        "trajectory": [[80, 0, 50, 1, 0, 0], [20, 0, 50, 1, 0, 0]],
    }
    selected = MODULE.select_distance_poses(row, [80, 40, 20])
    assert [item[1][0] for item in selected] == [0, 1]
