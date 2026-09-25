import numpy as np

from independent_nav.pointcloud import PerspectivePointRenderer, unproject_depth


def test_renderer_keeps_nearest_semantic_point():
    xyz = np.asarray([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    rgb = np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    semantic = np.asarray([1, 2], dtype=np.uint8)
    renderer = PerspectivePointRenderer(xyz, rgb, semantic, device="cpu")
    output = renderer.render([0, 0, 0, 1, 0, 0], width=16, height=16)
    assert output["valid"][8, 8]
    assert output["semantic"][8, 8] == 1
    assert output["rgb"][8, 8].tolist() == [255, 0, 0]
    assert np.isclose(output["depth"][8, 8], 5.0)


def test_renderer_marks_holes_invalid():
    renderer = PerspectivePointRenderer(
        np.asarray([[5.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[255, 0, 0]], dtype=np.uint8),
        np.asarray([1], dtype=np.uint8), device="cpu")
    output = renderer.render([0, 0, 0, 1, 0, 0], width=8, height=8)
    assert output["valid"].sum() == 1
    assert np.all(output["semantic"][~output["valid"]] == 255)


def test_depth_unprojection_recovers_world_point():
    depth = np.full((16, 16), np.nan, dtype=np.float32)
    depth[8, 8] = 5.0
    world = unproject_depth(depth, [0, 0, 0, 1, 0, 0])
    assert np.allclose(world[8, 8], [5.0, 0.0, 0.0])
