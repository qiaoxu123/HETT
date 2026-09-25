import numpy as np

from multiagent.active_altitude.controller import (
    AltitudeAction,
    ControllerConfig,
    NavigationPhase,
)
from multiagent.active_altitude.runtime import ActiveAltitudeRuntime
from multiagent.active_altitude.semantics import SemanticPrior


class CenterPriorProvider:
    def get(self, map_name, instruction_id, grid_shape, altitude_level):
        logits = np.zeros(grid_shape, dtype=np.float32)
        logits[grid_shape[0] // 2, grid_shape[1] // 2] = 8.0
        return SemanticPrior(logits, confidence=0.9, source="test")


def test_runtime_composes_semantics_belief_and_altitude_control():
    runtime = ActiveAltitudeRuntime(
        batch_size=1,
        grid_shape=(3, 3),
        config=ControllerConfig(transition_confirm_steps=1),
        semantic_provider=CenterPriorProvider(),
    )
    visual_logits = np.zeros((3, 3), dtype=np.float32)
    visual_logits[1, 1] = 10.0

    result = runtime.step(
        0,
        "map",
        "instruction",
        visual_logits,
        predicted_goal_xy=(0.5, 0.5),
        predicted_distance_m=80.0,
        stop_probability=0.0,
        landmark_confidence=1.0,
    )

    assert result.semantic_confidence == 0.9
    assert np.isclose(result.belief.sum(), 1.0)
    assert result.belief[1, 1] == result.belief.max()
    assert result.belief_diagnostics.observations == 1
    assert result.decision.phase == NavigationPhase.COARSE_APPROACH
    assert result.decision.altitude_action == AltitudeAction.DESCEND
    assert result.decision.target_altitude_m == 80.0


def test_runtime_keeps_episode_state_independent_within_batch():
    runtime = ActiveAltitudeRuntime(
        batch_size=2,
        grid_shape=(2, 2),
        config=ControllerConfig(
            transition_confirm_steps=1,
            phase_enter_confidence=(0.0, 0.0, 0.0, 0.0),
        ),
    )
    confident = np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float32)

    runtime.step(
        0, "map", "first", confident, (0.8, 0.8), 80.0, 0.0,
        landmark_confidence=1.0,
    )

    assert runtime.episodes[0].controller_state.phase == NavigationPhase.COARSE_APPROACH
    assert runtime.episodes[0].controller_state.current_level == 2
    assert runtime.episodes[1].controller_state.phase == NavigationPhase.SEARCH
    assert runtime.episodes[1].controller_state.current_level == 3
    assert runtime.episodes[1].belief.observations == 0
