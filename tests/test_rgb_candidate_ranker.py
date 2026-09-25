import json

import pytest
import torch

from scripts.train_rgb_candidate_ranker import (
    CandidateRanker,
    metrics,
    prepare_rows,
    landmark_geometry,
    same_map_switch_indices,
    validate_feature_alignment,
)


def test_candidate_ranker_returns_one_score_per_candidate():
    model = CandidateRanker(feature_dim=8, top_k=5)

    score = model(torch.randn(3, 5, 8), torch.randn(3, 8))

    assert score.shape == (3, 5)
    assert torch.isfinite(score).all()


def test_candidate_ranker_accepts_per_landmark_relation_context():
    model = CandidateRanker(feature_dim=8, top_k=5, relation_dim=6)

    score = model(
        torch.randn(3, 5, 8), torch.randn(3, 8), torch.randn(3, 5, 2, 10),
        torch.ones(3, 2), torch.randn(3, 2, 6))

    assert score.shape == (3, 5)
    assert torch.isfinite(score).all()


def test_candidate_metrics_select_requested_rank():
    distance = torch.tensor([[40.0, 10.0], [15.0, 70.0]])

    result = metrics(distance, torch.tensor([1, 0]))

    assert result["mean_ne"] == 12.5
    assert result["hit20"] == 100.0


def test_validation_rows_are_never_given_target_candidate(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {
        "map_name": "map_a",
        "target": [0.9, 0.9],
        "candidates": [[0.0, 0.0], [0.1, 0.1]],
    }
    path.write_text(json.dumps(row) + "\n")

    rows, injected = prepare_rows(path, 1, seed=0, top_k=2, inject_positive=False)

    assert injected == 0
    assert torch.allclose(torch.tensor(rows[0]["candidates"]),
                          torch.tensor(row["candidates"]))


def test_positive_injection_is_explicitly_training_only(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {
        "map_name": "map_a",
        "target": [0.9, 0.9],
        "candidates": [[0.0, 0.0], [0.1, 0.1]],
    }
    path.write_text(json.dumps(row) + "\n")

    rows, injected = prepare_rows(path, 1, seed=0, top_k=2, inject_positive=True)

    assert injected == 1
    candidates = torch.tensor(rows[0]["candidates"])
    target = torch.tensor(row["target"])
    assert torch.isclose(candidates, target).all(dim=-1).any()


def test_instruction_switch_stays_on_map_and_changes_instruction():
    rows = [
        {"map_name": "a", "description": "first"},
        {"map_name": "b", "description": "third"},
        {"map_name": "a", "description": "second"},
        {"map_name": "b", "description": "fourth"},
    ]

    switched = same_map_switch_indices(rows)

    for index, replacement in enumerate(switched.tolist()):
        assert rows[index]["map_name"] == rows[replacement]["map_name"]
        assert rows[index]["description"] != rows[replacement]["description"]


def test_landmark_geometry_uses_candidates_and_not_target():
    rows = [{
        "candidates": [[0.75, 0.5]],
        "target": [0.1, 0.1],
        "landmark_centers": [[0.5, 0.5]],
        "landmark_boundaries": [[[0.4, 0.5]] * 16],
    }]

    geometry, valid = landmark_geometry(rows, max_landmarks=1)

    assert geometry.shape == (1, 1, 1, 10)
    assert torch.allclose(geometry[0, 0, 0, :2], torch.tensor([0.25, 0.0]))
    assert torch.allclose(geometry[0, 0, 0, 5:7], torch.tensor([0.35, 0.0]))
    assert valid.tolist() == [[1.0]]


def test_feature_cache_alignment_rejects_reordered_rows():
    features = {"distance": torch.tensor([[41.0, 82.0]])}
    rows = [{"candidates": [[0.1, 0.0], [0.2, 0.0]], "target": [0.0, 0.0]}]
    validate_feature_alignment(features, rows)

    rows[0]["candidates"].reverse()
    with pytest.raises(ValueError, match="not aligned"):
        validate_feature_alignment(features, rows)
