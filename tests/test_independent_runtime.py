import inspect

import torch

from independent_nav.runtime import (
    DESCEND, OBSERVE, ActiveBeliefNavigator, ObservationEvidence,
    RuntimeConfig, RuntimePhase, ViewPlan,
)


def center_view(_belief, _xy, _altitude, _observations):
    return ViewPlan((0.5, 0.5), 35.0, 0.0)


def test_runtime_step_has_no_target_ground_truth_argument():
    parameters = inspect.signature(ActiveBeliefNavigator.step).parameters
    assert "target" not in parameters and "target_xy" not in parameters


def test_uncertain_belief_moves_to_active_view_and_changes_height():
    runtime = ActiveBeliefNavigator(torch.full((4, 4), 0.5), center_view)
    decision = runtime.step((0.5, 0.5), 0.0, 50.0)
    assert decision.phase == RuntimePhase.VERIFY
    assert decision.action == DESCEND
    decision = runtime.step((0.5, 0.5), 0.0, 35.0)
    assert decision.action == OBSERVE


def test_signed_observation_changes_goal_then_approaches():
    config = RuntimeConfig(maximum_observations=1, concentration_threshold=0.3)
    runtime = ActiveBeliefNavigator(torch.full((4, 4), 0.5), center_view, config)
    evidence = torch.full((4, 4), -3.0)
    evidence[3, 3] = 6.0
    observation = ObservationEvidence(
        evidence, torch.ones(4, 4, dtype=torch.bool), 1.0, "frame-1", 0.9, 0.9)
    decision = runtime.step((0.1, 0.1), 0.0, 50.0, observation)
    assert decision.phase == RuntimePhase.APPROACH
    assert decision.predicted_goal_xy == (0.875, 0.875)


def test_stop_requires_stability_visual_relation_and_two_confirmations():
    config = RuntimeConfig(
        maximum_observations=2, entropy_view_threshold=1.1,
        concentration_threshold=0.3, stable_goal_frames=2,
        stop_confirm_frames=2)
    prior = torch.full((4, 4), 0.01); prior[2, 2] = 0.99
    runtime = ActiveBeliefNavigator(prior, center_view, config)
    evidence = torch.zeros(4, 4)
    observation = ObservationEvidence(
        evidence, torch.ones(4, 4, dtype=torch.bool), 1.0, "one", 0.9, 0.9)
    first = runtime.step((0.625, 0.625), 0.0, 50.0, observation)
    second = runtime.step((0.625, 0.625), 0.0, 50.0)
    third = runtime.step((0.625, 0.625), 0.0, 50.0)
    assert not first.should_stop
    assert not second.should_stop
    assert third.should_stop
    assert third.phase == RuntimePhase.STOPPED
