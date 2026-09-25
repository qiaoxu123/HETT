import torch

from scripts.train_relation_geometry_ranker import (
    RelationGeometryRanker,
    audit_consistent_landmark_geometry,
    opposite_direction_cues,
    matched_same_map_switch_indices,
    ranking_loss,
    relation_cue_vector,
    rotate_270,
)


def test_rotate_270_reference_frame():
    delta = torch.tensor([[2.0, 3.0], [-4.0, 5.0]])
    assert torch.equal(rotate_270(delta), torch.tensor([[3.0, -2.0], [5.0, 4.0]]))


def test_audit_consistent_geometry_uses_transpose_transform():
    rows = [{
        "candidates": [[0.75, 0.5]],
        "landmark_centers": [[0.5, 0.5]],
        "landmark_boundaries": [[[0.4, 0.5]] * 16],
    }]
    geometry, valid = audit_consistent_landmark_geometry(rows, max_landmarks=1)
    assert torch.allclose(
        geometry[0, 0, 0, :2].float(), torch.tensor([0.0, 0.25]), atol=1e-3)
    assert valid.tolist() == [[True]]


def test_ranker_candidate_permutation_equivariance():
    torch.manual_seed(0)
    model = RelationGeometryRanker(hidden=32).eval()
    relation = torch.randn(2, 3, 768)
    geometry = torch.randn(2, 5, 3, 10)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    order = torch.tensor([3, 0, 4, 1, 2])
    with torch.inference_mode():
        expected = model(relation, geometry, valid)[:, order]
        actual = model(relation, geometry[:, order], valid)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_ranking_loss_prefers_near_candidate():
    distance = torch.tensor([[5.0, 80.0, 100.0]])
    good = torch.tensor([[4.0, -2.0, -3.0]])
    bad = torch.tensor([[-3.0, 4.0, -2.0]])
    assert ranking_loss(good, distance, 0.2)[0] < ranking_loss(bad, distance, 0.2)[0]


def test_local_relation_cues_and_opposites():
    cue = relation_cue_vector("Target is to the left of [MASK] and near the road")
    assert cue.sum() == 2
    swapped = opposite_direction_cues(cue)
    from scripts.train_relation_geometry_ranker import CUE_NAMES
    index = {name: i for i, name in enumerate(CUE_NAMES)}
    assert cue[index["left"]] == 1 and swapped[index["right"]] == 1
    assert swapped[index["left"]] == 0 and swapped[index["near"]] == 1


def test_switch_preserves_map_and_landmark_count():
    rows = [
        {"map_name": "a", "description": "one", "landmark_names": ["x"]},
        {"map_name": "a", "description": "two", "landmark_names": ["y"]},
        {"map_name": "a", "description": "three", "landmark_names": ["x", "y"]},
    ]
    switched = matched_same_map_switch_indices(rows)
    assert switched.tolist() == [1, 0, 2]
