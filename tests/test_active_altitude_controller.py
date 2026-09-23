import numpy as np

from multiagent.active_altitude.controller import (
    AltitudeAction,
    ControllerConfig,
    ControllerSignals,
    NavigationPhase,
    UncertaintyAltitudeController,
)


def signals(probs, xy=(0.5, 0.5), distance=120.0, stop=0.0,
            landmark=0.0, semantic=0.0):
    return ControllerSignals(probs, xy, distance, landmark, semantic, stop)


def test_controller_starts_high_and_descends_at_most_one_level_per_step():
    config = ControllerConfig(transition_confirm_steps=1)
    controller = UncertaintyAltitudeController(config)
    state = controller.initial_state()
    assert state.current_level == 3
    decision = controller.step(
        state, signals([0.01, 0.01, 0.01, 0.97], landmark=1.0, semantic=1.0))
    assert decision.phase == NavigationPhase.COARSE_APPROACH
    assert decision.altitude_action == AltitudeAction.DESCEND
    assert state.current_level == 2


def test_refinement_stop_requires_stability_distance_confidence_and_confirmation():
    config = ControllerConfig(transition_confirm_steps=1, stop_confirm_steps=2)
    controller = UncertaintyAltitudeController(config)
    state = controller.initial_state()
    state.phase = NavigationPhase.REFINEMENT
    state.current_level = 0
    first = controller.step(state, signals([0.01, 0.01, 0.01, 0.97], distance=10, stop=0.9))
    second = controller.step(state, signals([0.01, 0.01, 0.01, 0.97], xy=(0.51, 0.5), distance=10, stop=0.9))
    third = controller.step(state, signals([0.01, 0.01, 0.01, 0.97], xy=(0.505, 0.5), distance=10, stop=0.9))
    assert not first.should_stop
    assert not second.should_stop
    assert third.should_stop


def test_high_entropy_can_reopen_phase_and_ascend():
    config = ControllerConfig(recovery_confirm_steps=2)
    controller = UncertaintyAltitudeController(config)
    state = controller.initial_state()
    state.phase = NavigationPhase.LANDMARK_APPROACH
    state.current_level = 1
    uniform = np.ones(25) / 25
    controller.step(state, signals(uniform, distance=50))
    decision = controller.step(state, signals(uniform, distance=50))
    assert decision.phase == NavigationPhase.COARSE_APPROACH
    assert decision.altitude_action == AltitudeAction.ASCEND
