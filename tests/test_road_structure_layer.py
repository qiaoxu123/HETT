import numpy as np

from scripts.build_road_structure_layer import build_binary_layer


def test_road_layer_is_binary_and_excludes_non_roads(monkeypatch):
    class Bounds:
        x_min = 0.0
        y_max = 410.0

    monkeypatch.setitem(
        __import__("scripts.build_road_structure_layer", fromlist=["MAP_BOUNDS"]).MAP_BOUNDS,
        "map", Bounds())
    objects = {"map": {
        "road": {"object_type": "TrafficRoad",
                 "contour": [[0, 410], [205, 410], [205, 205], [0, 205]]},
        "building": {"object_type": "Building",
                     "contour": [[205, 205], [410, 205], [410, 0], [205, 0]]},
    }}

    mask, count = build_binary_layer(objects, "map", resolution=32)

    assert count == 1
    assert set(np.unique(mask).tolist()) == {0, 1}
    assert mask[4, 4] == 1
    assert mask[24, 24] == 0


def test_footpath_is_an_explicit_optional_channel(monkeypatch):
    class Bounds:
        x_min = 0.0
        y_max = 410.0

    monkeypatch.setitem(
        __import__("scripts.build_road_structure_layer", fromlist=["MAP_BOUNDS"]).MAP_BOUNDS,
        "map", Bounds())
    objects = {"map": {"path": {"object_type": "Footpath",
                                  "contour": [[0, 410], [410, 410],
                                              [410, 0], [0, 0]]}}}

    road_only, road_count = build_binary_layer(objects, "map", resolution=16)
    with_path, combined_count = build_binary_layer(
        objects, "map", resolution=16, object_types=("TrafficRoad", "Footpath"))

    assert road_count == 0
    assert road_only.sum() == 0
    assert combined_count == 1
    assert with_path.all()
