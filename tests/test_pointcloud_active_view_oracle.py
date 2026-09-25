import numpy as np

from scripts.evaluate_pointcloud_active_view_oracle import (
    eligible,
    generate_views,
)


def test_diagnostic_subset_requires_top5_success_and_top1_failure():
    def item(distances):
        return {
            "candidates": np.asarray([[value / 410.0, 0.0] for value in distances]),
            "target": np.zeros(2),
        }
    values = {
        "top1_correct": item([10, 30, 40, 50, 60]),
        "recoverable": item([30, 10, 40, 50, 60]),
        "unrecoverable": item([30, 40, 50, 60, 70]),
    }
    assert eligible(values) == [values["recoverable"]]


def test_cluster_views_span_directions_and_heights():
    candidates = np.asarray([[0.0, 0.0], [20.0, 0.0]])
    xx, yy = np.meshgrid(np.arange(-50, 71, 5), np.arange(-50, 51, 5))
    xyz = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    native = np.zeros(len(xyz), dtype=np.uint8)
    poses, descriptors = generate_views(
        candidates, xyz, native, heights=(20.0, 35.0, 50.0),
        directions=8, radius=30.0, view_mode="cluster")
    assert poses.shape == (24, 6)
    assert len(descriptors) == 24
    assert set(value[1] for value in descriptors) == {20.0, 35.0, 50.0}


def test_candidate_orbits_keep_all_candidate_anchors():
    candidates = np.asarray([[0.0, 0.0], [20.0, 0.0]])
    xyz = np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    native = np.zeros(len(xyz), dtype=np.uint8)
    poses, descriptors = generate_views(
        candidates, xyz, native, heights=(20.0,), directions=4,
        radius=30.0, view_mode="candidate_orbits")
    assert poses.shape == (8, 6)
    assert {value[0] for value in descriptors} == {0, 1}
