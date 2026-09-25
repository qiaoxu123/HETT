import math

import torch

from scripts.validate_candidate_geometry import (
    candidate_geometry,
    has_direction_relation,
    mask_landmark_span,
    polygon_group_centroid,
    resample_contour_group,
    resolve_landmark_group,
)


def test_candidate_geometry_uses_north_up_continuous_angles():
    candidates = torch.tensor([[0.75, 0.50], [0.50, 0.25]])
    landmarks = torch.tensor([[[0.50, 0.50]]])

    features = candidate_geometry(candidates, landmarks)[0, 0]

    # Image-normalized y grows southward: east is (cos=1, sin=0), north is
    # (cos=0, sin=-1). The continuous representation has no angle wraparound.
    assert torch.allclose(features[0, :2], torch.tensor([0.25, 0.0]))
    assert torch.allclose(features[0, 3:], torch.tensor([0.0, 1.0]), atol=1e-5)
    assert torch.allclose(features[1, 3:], torch.tensor([-1.0, 0.0]), atol=1e-5)
    expected = math.log1p(0.25 * 410.0) / math.log1p(410.0)
    assert torch.isclose(features[0, 2], torch.tensor(expected), atol=1e-5)


def test_candidate_geometry_is_finite_at_landmark_center():
    candidates = torch.tensor([[0.5, 0.5]])
    landmarks = torch.tensor([[[0.5, 0.5]]])

    features = candidate_geometry(candidates, landmarks)

    assert torch.isfinite(features).all()


def test_boundary_geometry_uses_nearest_contour_point():
    candidates = torch.tensor([[0.8, 0.5]])
    centers = torch.tensor([[[0.5, 0.5]]])
    boundary = torch.tensor([[[[0.4, 0.5], [0.6, 0.5]]]])

    features = candidate_geometry(candidates, centers, boundary)

    assert features.shape[-1] == 10
    assert torch.allclose(features[0, 0, 0, 5:7], torch.tensor([0.2, 0.0]), atol=1e-5)


def test_landmark_span_mask_tolerates_punctuation_variation():
    text = "The building behind St Teresa of the Child Jesus Catholic Church."
    query = "St. Teresa of the Child Jesus Catholic Church"

    masked, found = mask_landmark_span(text, query)

    assert found
    assert "[MASK]" in masked
    assert "behind [MASK]" in masked


def test_landmark_span_mask_uses_fuzzy_alias_without_retaining_name():
    text = "The house faces Wellington street between Leslie and Willmore."

    masked, found = mask_landmark_span(text, "Wellington Road")

    assert found
    assert "[MASK]" in masked
    assert "Wellington" not in masked


def test_direction_relation_detection_ignores_near_only_language():
    assert has_direction_relation("A building north of the church")
    assert has_direction_relation("A car across from the station")
    assert not has_direction_relation("A building near the church")


def test_exact_duplicate_landmark_returns_all_components():
    candidates = [
        {"name": "Walsall Road", "contour": [[0, 0]]},
        {"name": "Walsall Road", "contour": [[10, 0]]},
        {"name": "Other Road", "contour": [[20, 0]]},
    ]

    result = resolve_landmark_group("Walsall Road", candidates)

    assert len(result) == 2


def test_uncertain_fuzzy_landmark_is_rejected():
    candidates = [{"name": "The Museum"}, {"name": "Main Road"}]

    assert resolve_landmark_group("The Quadrangle", candidates) == []


def test_group_geometry_covers_disconnected_components():
    left = [[0, 0], [2, 0], [2, 2], [0, 2]]
    right = [[8, 0], [10, 0], [10, 2], [8, 2]]

    center = polygon_group_centroid([left, right])
    boundary = resample_contour_group([left, right], count=16)

    assert torch.allclose(torch.tensor(center), torch.tensor([5.0, 1.0]))
    assert (boundary[:, 0] < 3).any()
    assert (boundary[:, 0] > 7).any()
