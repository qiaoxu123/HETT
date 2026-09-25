import numpy as np
import pytest

from independent_nav.relation_spatial_field import (
    directional_field,
    extract_relation,
    mask_landmark_span,
    near_field,
)


BOUNDARY = np.asarray(
    [[0.45, 0.45], [0.55, 0.45], [0.55, 0.55], [0.45, 0.55]],
    dtype=np.float32,
)


def test_strict_relation_is_bound_to_masked_landmark():
    masked, found = mask_landmark_span(
        "A car to the left of St Mary's Church.",
        "St Mary's Church",
        "St Mary's Church",
    )
    assert found
    assert extract_relation(masked) == {"directions": ["left"], "near": False}


def test_unrelated_near_word_is_not_bound_to_landmark():
    parsed = extract_relation(
        "A building between [MASK] and the road, with cars parked near it."
    )
    assert not parsed["near"]


def test_near_relation_is_bound_to_landmark():
    parsed = extract_relation("A white car in the parking lot near [MASK].")
    assert parsed["near"]


def test_fixed_rotation_directional_field_and_flip():
    # Under the audited rotate_270 frame, language-left maps down in normalized
    # map coordinates, while language-right maps up.
    points = np.asarray([[0.50, 0.70], [0.50, 0.30]], dtype=np.float32)
    left = directional_field(points, BOUNDARY, "left")
    right = directional_field(points, BOUNDARY, "right")
    assert left[0] > left[1]
    assert right[0] < right[1]


def test_near_uses_boundary_not_landmark_center():
    near_large_boundary = np.asarray([[0.1, 0.5], [0.9, 0.5]], dtype=np.float32)
    points = np.asarray([[0.91, 0.5], [0.5, 0.8]], dtype=np.float32)
    scores = near_field(points, near_large_boundary, scale_m=30.0)
    assert scores[0] > scores[1]


def test_invalid_near_scale_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        near_field(np.asarray([[0.5, 0.5]]), BOUNDARY, scale_m=0)
