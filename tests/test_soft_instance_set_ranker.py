import copy

import torch

from scripts.train_soft_instance_set_ranker import (
    SoftInstanceSetRanker, build_component_set, relation_geometry,
)


def sample_row():
    boundary = [[0.40, 0.45], [0.60, 0.45], [0.60, 0.55], [0.40, 0.55]] * 4
    return {
        "map_name": "map",
        "candidates": [[0.48, 0.50], [0.70, 0.50], [0.50, 0.70],
                       [0.30, 0.50], [0.50, 0.30]],
        "target": [0.99, 0.99],
        "landmark_centers": [[0.50, 0.50]],
        "landmark_boundaries": [boundary],
    }


def components():
    contour = torch.tensor([[0.40, 0.45], [0.50, 0.45], [0.50, 0.55],
                            [0.40, 0.55]] * 4)
    item = (torch.tensor([0.45, 0.50]), contour)
    return {"map": [[item], [], [], []]}


def test_soft_component_set_does_not_read_target_coordinate():
    first = sample_row()
    second = copy.deepcopy(first)
    second["target"] = [0.01, 0.01]
    probability = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    a = build_component_set([first], probability, components())
    b = build_component_set([second], probability, components())
    for left, right in zip(a[:5], b[:5]):
        assert torch.equal(left, right)


def test_soft_set_model_returns_finite_candidate_scores():
    row = sample_row()
    probability = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    centers, boundaries, valid, prior, candidates, _ = build_component_set(
        [row], probability, components())
    geometry, landmark_valid = relation_geometry(
        [row], candidates, centers, boundaries)
    model = SoftInstanceSetRanker(geometry.shape[-1])
    score = model(torch.zeros(1, 9, 768), geometry, landmark_valid, valid,
                  prior, torch.zeros(1, 9, 12))
    assert score.shape == (1, 5)
    assert score.isfinite().all()
