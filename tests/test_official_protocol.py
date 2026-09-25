import inspect

import pytest
import torch

from independent_nav.official_protocol import (
    OfficialProtocol, belief_argmax_xy, canonical_instruction, optimistic_translation_steps,
    generate_budgeted_waypoints, route_fits_budget, rollout_static_goals,
)


def test_protocol_is_frozen_to_paper_settings():
    OfficialProtocol().validate()
    with pytest.raises(ValueError):
        OfficialProtocol(max_action_len=21).validate()
    with pytest.raises(ValueError):
        OfficialProtocol(altitude_m=35).validate()


def test_belief_argmax_uses_xy_not_row_column_order():
    logits = torch.zeros(4, 4)
    logits[1, 3] = 1
    assert belief_argmax_xy(logits) == (0.875, 0.375)


def test_budget_filter_counts_route_via_observation_point():
    protocol = OfficialProtocol()
    assert optimistic_translation_steps((0, 0), (0.2, 0), protocol) == 2
    assert route_fits_budget((0, 0), (0.1, 0), (0.2, 0), 2, protocol)
    assert not route_fits_budget((0, 0), (0.5, 0.5), (0.2, 0), 2, protocol)


def test_instruction_identity_ignores_processing_only_punctuation():
    assert canonical_instruction('a flipped "L"') == canonical_instruction(
        "A  flipped L")


def test_rollout_policy_has_no_target_argument():
    parameters = inspect.signature(rollout_static_goals).parameters
    assert "target" not in parameters and "target_xy" not in parameters


def test_path_view_candidates_are_one_action_and_budget_constrained():
    protocol = OfficialProtocol()
    current, top1, top2 = (0.1, 0.1), (0.8, 0.1), (0.8, 0.3)
    candidates = generate_budgeted_waypoints(
        current, top1, top2, remaining_actions=8, protocol=protocol)
    assert {item.name for item in candidates} >= {
        "toward_top1", "toward_midpoint", "path_left", "path_right"}
    maximum = protocol.maximum_translation_per_action_m + 1e-6
    for item in candidates:
        distance = torch.linalg.vector_norm(
            torch.tensor(item.xy) - torch.tensor(current)).item() * 410
        assert distance <= maximum
        assert route_fits_budget(
            current, item.xy, top1, 8, protocol)


def test_tight_budget_removes_detours():
    protocol = OfficialProtocol()
    candidates = generate_budgeted_waypoints(
        (0.1, 0.1), (0.8, 0.1), (0.8, 0.8),
        remaining_actions=6, protocol=protocol)
    assert all(item.name != "toward_top2" for item in candidates)
