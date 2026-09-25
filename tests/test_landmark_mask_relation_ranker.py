import copy

import torch

from scripts.train_landmark_mask_relation_ranker import shape_aware_geometry


def row():
    boundary = [[0.40, 0.45], [0.60, 0.45], [0.60, 0.55], [0.40, 0.55]] * 4
    return {
        "candidates": [[0.50, 0.50], [0.70, 0.50], [0.50, 0.70],
                       [0.30, 0.50], [0.50, 0.30]],
        "target": [0.51, 0.51],
        "landmark_centers": [[0.50, 0.50]],
        "landmark_boundaries": [boundary],
    }


def test_shape_geometry_does_not_read_target_coordinate():
    first = row()
    second = copy.deepcopy(first)
    second["target"] = [0.99, 0.01]
    a, _ = shape_aware_geometry([first])
    b, _ = shape_aware_geometry([second])
    assert torch.equal(a, b)


def test_new_shape_support_is_invariant_to_contour_start_point():
    first = row()
    second = copy.deepcopy(first)
    contour = second["landmark_boundaries"][0]
    second["landmark_boundaries"][0] = contour[3:] + contour[:3]
    a, _ = shape_aware_geometry([first])
    b, _ = shape_aware_geometry([second])
    # The first ten values deliberately retain the legacy nearest-point
    # descriptor, whose direction is ambiguous under an exact distance tie.
    # The newly added full-mask support and shape values must be invariant.
    assert torch.allclose(a[..., 10:], b[..., 10:], atol=1e-3)


def test_shape_geometry_retains_oriented_landmark_extent():
    horizontal = row()
    vertical = copy.deepcopy(horizontal)
    vertical["landmark_boundaries"][0] = [
        [0.45, 0.40], [0.55, 0.40], [0.55, 0.60], [0.45, 0.60]
    ] * 4
    a, _ = shape_aware_geometry([horizontal])
    b, _ = shape_aware_geometry([vertical])
    assert not torch.allclose(a, b)
