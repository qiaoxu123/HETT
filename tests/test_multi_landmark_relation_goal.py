from pathlib import Path

import torch

from multiagent.models.relation_goal import (
    RelationGoalRanker,
    candidate_landmark_geometry,
    merge_goal_logits,
)
from multiagent.multi_landmark import MultiLandmarkIndex


def test_geometry_keeps_landmarks_as_separate_axis():
    candidates = torch.tensor([[[0.5, 0.5], [0.8, 0.8]]])
    centers = torch.tensor([[[0.2, 0.5], [0.5, 0.2]]])
    boundaries = centers[:, :, None].expand(-1, -1, 16, -1).clone()
    valid = torch.tensor([[True, True]])
    geometry = candidate_landmark_geometry(candidates, centers, boundaries, valid)
    assert geometry.shape == (1, 2, 2, 10)
    assert not torch.allclose(geometry[:, :, 0], geometry[:, :, 1])


def test_relation_ranker_is_candidate_permutation_equivariant():
    torch.manual_seed(0)
    model = RelationGoalRanker().eval()
    candidates = torch.rand(2, 7, 2)
    centers = torch.rand(2, 3, 2)
    boundaries = torch.rand(2, 3, 16, 2)
    valid = torch.ones(2, 3, dtype=torch.bool)
    relation = torch.rand(2, 3, 768)
    cue = torch.rand(2, 3, 12)
    order = torch.tensor([3, 1, 6, 0, 4, 2, 5])
    normal = model(candidates, centers, boundaries, valid, relation, cue)
    permuted = model(candidates[:, order], centers, boundaries, valid, relation, cue)
    assert torch.allclose(permuted, normal[:, order], atol=1e-5)


def test_relation_only_removes_hett_goal_prior():
    relation = torch.tensor([[1.0, -1.0, 0.5]])
    first_hett = torch.tensor([[100.0, 0.0, 0.0]])
    second_hett = torch.tensor([[0.0, 100.0, 0.0]])
    first = merge_goal_logits(first_hett, relation, 0.5, relation_only=True)
    second = merge_goal_logits(second_hett, relation, 0.5, relation_only=True)
    assert torch.equal(first, relation)
    assert torch.equal(second, relation)


def test_exported_feature_artifact_has_independent_landmarks():
    path = Path("runs/multi_landmark_v1/features.pt")
    if not path.exists():
        return
    index = MultiLandmarkIndex(path)
    key = next(key for key, row in index.rows.items() if int(row[2].sum()) >= 2)
    batch = index.batch([key], torch.device("cpu"))
    assert batch.centers.shape == (1, 9, 2)
    assert batch.boundaries.shape == (1, 9, 16, 2)
    assert batch.relation.shape == (1, 9, 768)
    assert batch.valid[0].sum() >= 2
    assert not torch.equal(batch.relation[0, 0], batch.relation[0, 1])


def test_wrong_controls_change_only_the_requested_fields(tmp_path):
    centers = torch.tensor([
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.6, 0.7], [0.8, 0.9]],
    ])
    boundaries = centers[:, :, None].expand(-1, -1, 16, -1).clone()
    valid = torch.ones(2, 2, dtype=torch.bool)
    relation = torch.stack((torch.zeros(2, 768), torch.ones(2, 768)))
    cue = torch.stack((torch.zeros(2, 12), torch.ones(2, 12)))
    path = tmp_path / "features.pt"
    torch.save({
        "schema": 1,
        "max_landmarks": 2,
        "splits": {"val_unseen": {
            "episode_ids": [["map", "object-a", 0], ["map", "object-b", 1]],
            "centers": centers,
            "boundaries": boundaries,
            "valid": valid,
            "relation": relation,
            "cue": cue,
        }},
    }, path)
    index = MultiLandmarkIndex(path)
    episode = [["map", "object-a", 0]]
    normal = index.batch(episode, torch.device("cpu"))
    wrong_relation = index.batch(
        episode, torch.device("cpu"), control="wrong_relation")
    wrong_landmark = index.batch(
        episode, torch.device("cpu"), control="wrong_landmark")
    assert torch.equal(normal.centers, wrong_relation.centers)
    assert not torch.equal(normal.relation, wrong_relation.relation)
    assert not torch.equal(normal.centers, wrong_landmark.centers)
    assert torch.equal(normal.relation, wrong_landmark.relation)
