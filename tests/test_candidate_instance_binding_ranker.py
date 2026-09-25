import copy

import torch

from scripts.train_candidate_instance_binding_ranker import (
    bind_candidate_instances, pair_geometry,
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


def sample_components():
    first = (torch.tensor([0.45, 0.50]),
             torch.tensor([[0.40, 0.45], [0.50, 0.45], [0.50, 0.55],
                           [0.40, 0.55]] * 4))
    second = (torch.tensor([0.75, 0.50]),
              torch.tensor([[0.70, 0.45], [0.80, 0.45], [0.80, 0.55],
                            [0.70, 0.55]] * 4))
    return {"map": [[first, second], [], [], []]}


def test_binding_uses_predicted_class_and_candidate_not_target():
    first = sample_row()
    second = copy.deepcopy(first)
    second["target"] = [0.01, 0.01]
    a = bind_candidate_instances([first], torch.tensor([0]), sample_components())
    b = bind_candidate_instances([second], torch.tensor([0]), sample_components())
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_instance_landmark_pair_geometry_is_finite_and_candidate_specific():
    row = sample_row()
    centers, boundaries, valid, _ = bind_candidate_instances(
        [row], torch.tensor([0]), sample_components())
    geometry, landmark_valid = pair_geometry([row], centers, boundaries, valid)
    assert geometry.shape == (1, 5, 9, 18)
    assert geometry.isfinite().all()
    assert landmark_valid[0, 0]
    assert not torch.equal(geometry[0, 0, 0], geometry[0, 1, 0])
