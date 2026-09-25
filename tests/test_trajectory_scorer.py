import torch

from independent_nav.trajectory_scorer import (
    MultiObjectiveTrajectoryScorer, TrajectoryHypothesisScorer,
)


def test_trajectory_scorer_is_candidate_equivariant_and_landmark_invariant():
    torch.manual_seed(4)
    model = TrajectoryHypothesisScorer(
        relation_dim=16, cue_dim=3, hidden=16, dropout=0).eval()
    values = {
        "relation": torch.randn(2, 3, 16),
        "cue": torch.randn(2, 3, 3),
        "geometry": torch.randn(2, 4, 3, 10),
        "landmark_valid": torch.tensor([[1, 1, 0], [1, 1, 1]]).bool(),
        "trajectory": torch.randn(2, 4, 6, 5),
        "belief_features": torch.randn(2, 4, 4),
    }
    values["trajectory"][..., -1] = 1
    with torch.inference_mode():
        expected = model(**values)
        candidate_order = torch.tensor([2, 0, 3, 1])
        candidate = dict(values)
        for key in ("geometry", "trajectory", "belief_features"):
            candidate[key] = candidate[key][:, candidate_order]
        landmark_order = torch.tensor([2, 0, 1])
        landmark = dict(values)
        for key in ("relation", "cue", "landmark_valid"):
            landmark[key] = landmark[key][:, landmark_order]
        landmark["geometry"] = landmark["geometry"][:, :, landmark_order]
        actual_candidate = model(**candidate)
        actual_landmark = model(**landmark)
    assert torch.allclose(
        actual_candidate, expected[:, candidate_order], atol=1e-5)
    assert torch.allclose(actual_landmark, expected, atol=1e-5)


def test_multi_objective_scorer_exposes_separate_components():
    model = MultiObjectiveTrajectoryScorer(
        relation_dim=16, cue_dim=3, route_channels=5,
        hidden=16, dropout=0).eval()
    output = model(
        torch.randn(1, 2, 16), torch.randn(1, 2, 3),
        torch.randn(1, 3, 2, 10), torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, 3, 4, 5), torch.randn(1, 3, 4),
        return_components=True)
    assert set(output) == {"final", "goal", "utility"}
    assert output["final"].shape == (1, 3)
