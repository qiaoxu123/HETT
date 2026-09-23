"""Rollout-facing composition of belief, semantics and control state."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .belief import BeliefDiagnostics, MultiScaleBelief
from .controller import (
    ControllerConfig,
    ControllerDecision,
    ControllerSignals,
    ControllerState,
    UncertaintyAltitudeController,
)
from .semantics import SemanticPriorProvider, NullSemanticPrior, gated_semantic_logits


@dataclass
class EpisodeRuntime:
    belief: MultiScaleBelief
    controller_state: ControllerState


@dataclass(frozen=True)
class RuntimeStep:
    decision: ControllerDecision
    belief: np.ndarray
    belief_diagnostics: BeliefDiagnostics
    semantic_confidence: float


class ActiveAltitudeRuntime:
    """Own per-episode state; never shares phase state across a batch."""

    def __init__(self, batch_size: int, grid_shape: tuple[int, int],
                 config: ControllerConfig = ControllerConfig(),
                 semantic_provider: SemanticPriorProvider = NullSemanticPrior()):
        self.grid_shape = grid_shape
        self.controller = UncertaintyAltitudeController(config)
        self.semantic_provider = semantic_provider
        self.episodes = [
            EpisodeRuntime(MultiScaleBelief(grid_shape), self.controller.initial_state())
            for _ in range(batch_size)
        ]

    def step(self, index: int, map_name: str, instruction_id: str,
             spatial_logits: Sequence[float], predicted_goal_xy: tuple[float, float],
             predicted_distance_m: float, stop_probability: float,
             landmark_confidence: float = 0.0) -> RuntimeStep:
        episode = self.episodes[index]
        altitude_level = episode.controller_state.current_level
        prior = self.semantic_provider.get(
            map_name, instruction_id, self.grid_shape, altitude_level)
        semantic_logits, semantic_confidence = gated_semantic_logits(prior, self.grid_shape)
        altitude_m = self.controller.config.altitude_levels_m[altitude_level]
        belief = episode.belief.update(
            spatial_logits, altitude_m, semantic_logits=semantic_logits)
        diagnostics = episode.belief.diagnostics()
        decision = self.controller.step(
            episode.controller_state,
            ControllerSignals(
                goal_probabilities=belief.reshape(-1),
                predicted_goal_xy=predicted_goal_xy,
                predicted_distance_m=predicted_distance_m,
                landmark_confidence=landmark_confidence,
                semantic_confidence=semantic_confidence,
                stop_probability=stop_probability,
            ),
        )
        return RuntimeStep(decision, belief, diagnostics, semantic_confidence)
