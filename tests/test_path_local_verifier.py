import torch
import inspect

from independent_nav.path_local_verifier import (
    PathLocalCandidateVerifier, soft_distance_ranking_loss,
)
from scripts.evaluate_candidate_verifier_closed_loop import deployment_plans
from scripts.evaluate_online_candidate_verifier_closed_loop import (
    inference_plans as online_inference_plans, rollout_online,
)


def inputs():
    torch.manual_seed(0)
    return {
        "image": torch.randn(3, 2, 16),
        "text": torch.randn(3, 16),
        "geometry": torch.randn(3, 2, 4, 10),
        "relation": torch.randn(3, 4, 8),
        "landmark_valid": torch.tensor([
            [1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool),
        "prior": torch.rand(3, 2),
        "semantic_evidence": torch.rand(3, 2),
    }


def test_candidate_order_is_permutation_equivariant():
    model = PathLocalCandidateVerifier(16, relation_dim=8, hidden=16).eval()
    values = inputs()
    with torch.inference_mode():
        expected = model(**values)
        swapped = {key: value[:, [1, 0]] if key in {
            "image", "geometry", "prior", "semantic_evidence"} else value
                   for key, value in values.items()}
        actual = model(**swapped)
    assert torch.allclose(actual, expected[:, [1, 0]], atol=1e-6)


def test_loss_prefers_score_for_nearer_candidate():
    distance = torch.tensor([[5.0, 80.0], [70.0, 10.0]])
    correct = torch.tensor([[3.0, -1.0], [-1.0, 3.0]])
    wrong = correct[:, [1, 0]]
    assert soft_distance_ranking_loss(correct, distance) < soft_distance_ranking_loss(
        wrong, distance)


def test_deployment_plan_strips_metric_target():
    records = [{
        "modes": [[0.1, 0.2], [0.3, 0.4]],
        "observation_action_step": 2, "target": [0.9, 0.9],
    }]
    plan = deployment_plans(records, [1])[0]
    assert "target" not in plan
    assert "target" not in inspect.signature(deployment_plans).parameters


def test_online_verifier_plan_and_rollout_exclude_target():
    records = [{
        "modes": [[0.1, 0.2], [0.3, 0.4]],
        "observation_action_step": 2, "target": [0.9, 0.9],
    }]
    features = {
        "prior": torch.rand(1, 2), "text": torch.rand(1, 16),
        "geometry": torch.rand(1, 2, 3, 10),
        "relation": torch.rand(1, 3, 8),
        "valid": torch.ones(1, 3, dtype=torch.bool),
    }
    plan = online_inference_plans(records, features, [1], [0])[0]
    assert "target" not in plan
    assert "target" not in inspect.signature(rollout_online).parameters
