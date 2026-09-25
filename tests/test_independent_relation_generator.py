import torch

from multiagent.multi_landmark import IndependentGoalIndex
from scripts.evaluate_independent_relation_generator import summarize


def test_summary_uses_target_only_after_candidate_rank_is_fixed():
    distance = torch.tensor([[10.0, 30.0], [40.0, 15.0]])
    rank = torch.tensor([0, 1])
    count = torch.tensor([1, 2])
    result = summarize(distance, rank, count)
    assert result["hit20"] == 100.0
    assert result["mean_ne"] == 12.5
    assert result["multi_landmark"]["count"] == 1


def test_independent_goal_artifact_lookup(tmp_path):
    path = tmp_path / "goals.pt"
    torch.save({
        "schema": 1,
        "episode_ids": [["map", "object", 3]],
        "goals": torch.tensor([[0.25, 0.75]]),
    }, path)
    index = IndependentGoalIndex(path)
    result = index.batch([["map", "object", 3]], torch.device("cpu"))
    assert torch.equal(result, torch.tensor([[0.25, 0.75]]))
