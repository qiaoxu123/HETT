import numpy as np

from multiagent.visual_goal.abstraction import VisualMasks, make_levels, controlled_ablation
from multiagent.visual_goal.template_builder import polygon_mask, world_to_template


def _masks(n=32):
    target = np.zeros((n, n), bool); target[13:19, 13:19] = True
    anchor = np.zeros_like(target); anchor[5:10, 5:10] = True
    building = np.zeros_like(target); building[2:28, 2:28] = True
    road = np.zeros_like(target); road[27:30, :] = True
    parking = np.zeros_like(target); parking[3:7, 20:27] = True
    vegetation = np.zeros_like(target); vegetation[21:26, 4:9] = True
    open_area = ~(building | road | parking | vegetation)
    return VisualMasks(target, anchor, building, road, parking, vegetation, open_area)


def test_progressive_levels_have_fixed_canvas_and_named_semantics():
    image = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    out = make_levels(image, _masks())
    assert list(out) == [f"L{i}" for i in range(10)]
    assert all(x.shape == image.shape and x.dtype == np.uint8 for x in out.values())
    assert np.array_equal(out["L0"], image)
    assert not np.array_equal(out["L9"], image)


def test_target_anchor_context_controls_are_separate():
    image = np.full((32, 32, 3), 160, np.uint8)
    out = controlled_ablation(image, _masks())
    for key in ("target_only", "anchor_only", "target_anchor", "target_anchor_context",
                "color_gray", "texture_strong_blur", "geometry_contour"):
        assert out[key].shape == image.shape
    assert np.all(out["target_only"][~_masks().target] == 112)
    assert np.all(out["anchor_only"][~_masks().anchor] == 112)


def test_world_xy_maps_to_square_canvas_and_polygon():
    xy = np.asarray([[0, 10], [10, 0], [5, 5]], dtype=float)
    pix = world_to_template(xy, (5, 5), 10, 101)
    assert np.allclose(pix, [[100, 0], [0, 100], [50, 50]])
    mask = polygon_mask([[2, 8], [8, 8], [8, 2], [2, 2]], (5, 5), 10, 101)
    assert mask[50, 50] and mask.sum() > 3000
