import numpy as np

from multiagent.active_altitude.controller import AltitudeAction
from multiagent.active_altitude.fine_refinement import (
    FineAltitudeConfig,
    FineAltitudeController,
    FineAltitudeSignals,
    FineNavigationPhase,
    canonical_map_pose,
)
from multiagent.space import Pose4D


def confident(distance, xy=(0.5, 0.5)):
    probabilities = np.full(25, 0.001, dtype=np.float64)
    probabilities[-1] = 1.0
    probabilities /= probabilities.sum()
    return FineAltitudeSignals(probabilities, xy, distance)


def uncertain(distance, xy=(0.5, 0.5)):
    return FineAltitudeSignals(np.ones(25) / 25, xy, distance)


def test_coarse_navigation_stays_at_canonical_altitude_until_goal_is_stable():
    controller = FineAltitudeController(FineAltitudeConfig(transition_confirm_steps=1))
    state = controller.initial_state()
    for xy in ((0.2, 0.2), (0.5, 0.5), (0.8, 0.8)):
        decision = controller.step(state, confident(20, xy))
        assert decision.phase == FineNavigationPhase.COARSE
        assert decision.target_altitude_m == 50.0
        assert decision.altitude_action == AltitudeAction.HOLD


def test_refinement_descends_only_when_next_view_contains_stable_goal():
    controller = FineAltitudeController(FineAltitudeConfig(transition_confirm_steps=1))
    state = controller.initial_state()
    for _ in range(2):
        controller.step(state, confident(30))
    to_40 = controller.step(state, confident(30))
    assert to_40.phase == FineNavigationPhase.REFINE
    assert to_40.target_altitude_m == 40.0
    assert to_40.altitude_action == AltitudeAction.DESCEND

    # At 28 m the predicted target is outside the configured 30 m view margin.
    hold_40 = controller.step(state, confident(28))
    assert not hold_40.contained_by_next_view
    assert hold_40.target_altitude_m == 40.0

    to_30 = controller.step(state, confident(20))
    assert to_30.target_altitude_m == 30.0
    assert to_30.altitude_action == AltitudeAction.DESCEND
    assert not to_30.global_memory_eligible


def test_uncertainty_recovers_one_level_at_a_time_back_to_canonical_height():
    config = FineAltitudeConfig(transition_confirm_steps=1, recovery_confirm_steps=1)
    controller = FineAltitudeController(config)
    state = controller.initial_state()
    for _ in range(4):
        controller.step(state, confident(20))
    assert state.current_level == 2

    to_40 = controller.step(state, uncertain(20))
    assert to_40.altitude_action == AltitudeAction.ASCEND
    assert to_40.target_altitude_m == 40.0
    to_50 = controller.step(state, uncertain(20))
    assert to_50.altitude_action == AltitudeAction.ASCEND
    assert to_50.target_altitude_m == 50.0
    assert to_50.phase == FineNavigationPhase.COARSE
    assert to_50.global_memory_eligible


def test_canonical_map_pose_changes_only_height():
    camera_pose = Pose4D(10.0, 20.0, 130.0, 0.25)
    map_pose = canonical_map_pose(camera_pose, ground_level=100.0)
    assert map_pose == Pose4D(10.0, 20.0, 150.0, 0.25)
