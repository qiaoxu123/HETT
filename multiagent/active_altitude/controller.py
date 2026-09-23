"""Deployable uncertainty-guided altitude and stop controller.

This module never reads target ground truth.  Privileged supervision belongs in
``teacher.py`` so evaluation cannot accidentally leak oracle information.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from math import log
from typing import Sequence, Tuple

import numpy as np


class NavigationPhase(IntEnum):
    SEARCH = 0
    COARSE_APPROACH = 1
    LANDMARK_APPROACH = 2
    REFINEMENT = 3


class AltitudeAction(IntEnum):
    ASCEND = 0
    HOLD = 1
    DESCEND = 2


@dataclass(frozen=True)
class ControllerConfig:
    """Controller thresholds; altitude levels must be low-to-high AGL."""

    altitude_levels_m: Tuple[float, ...] = (30.0, 60.0, 80.0, 120.0)
    phase_step_m: Tuple[float, ...] = (50.0, 20.0, 10.0, 5.0)
    phase_enter_confidence: Tuple[float, ...] = (0.0, 0.42, 0.57, 0.70)
    phase_target_level: Tuple[int, ...] = (3, 2, 1, 0)
    stable_goal_radius: float = 0.035
    stable_goal_window: int = 3
    transition_confirm_steps: int = 2
    recovery_confirm_steps: int = 2
    recovery_entropy: float = 0.82
    refinement_distance_m: float = 40.0
    stop_distance_m: float = 20.0
    stop_probability: float = 0.70
    stop_confirm_steps: int = 2
    allow_recovery_ascent: bool = True

    def __post_init__(self):
        if len(self.altitude_levels_m) != 4:
            raise ValueError("the initial design requires four altitude levels")
        if tuple(sorted(self.altitude_levels_m)) != self.altitude_levels_m:
            raise ValueError("altitude levels must be strictly low-to-high")
        if any(b <= a for a, b in zip(self.altitude_levels_m, self.altitude_levels_m[1:])):
            raise ValueError("altitude levels must be unique")
        if not (len(self.phase_step_m) == len(self.phase_enter_confidence)
                == len(self.phase_target_level) == len(NavigationPhase)):
            raise ValueError("every navigation phase needs step, threshold and altitude")
        if self.stable_goal_window < 2 or self.transition_confirm_steps < 1:
            raise ValueError("invalid temporal confirmation configuration")


@dataclass(frozen=True)
class ControllerSignals:
    """Prediction-only signals available during deployment."""

    goal_probabilities: Sequence[float]
    predicted_goal_xy: Tuple[float, float]
    predicted_distance_m: float
    landmark_confidence: float = 0.0
    semantic_confidence: float = 0.0
    stop_probability: float = 0.0


@dataclass
class ControllerState:
    current_level: int = 3
    phase: NavigationPhase = NavigationPhase.SEARCH
    phase_evidence_steps: int = 0
    recovery_steps: int = 0
    stop_evidence_steps: int = 0
    goal_history: list[Tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class ControllerDecision:
    phase: NavigationPhase
    altitude_action: AltitudeAction
    target_level: int
    target_altitude_m: float
    horizontal_step_m: float
    should_stop: bool
    confidence: float
    normalized_entropy: float
    top2_margin: float
    stable_goal: bool
    recovery: bool


def _distribution_diagnostics(probabilities: Sequence[float]) -> tuple[float, float]:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size < 2 or not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("goal_probabilities must be a finite non-negative distribution")
    total = probs.sum()
    if total <= 0:
        raise ValueError("goal_probabilities must have positive mass")
    probs = probs / total
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None)))) / log(probs.size)
    top2 = np.partition(probs, -2)[-2:]
    margin = float(top2.max() - top2.min())
    return float(np.clip(entropy, 0.0, 1.0)), margin


def _goal_is_stable(history: Sequence[Tuple[float, float]], radius: float) -> bool:
    if len(history) < 2:
        return False
    points = np.asarray(history, dtype=np.float64)
    center = points.mean(axis=0)
    return bool(np.linalg.norm(points - center, axis=1).max() <= radius)


class UncertaintyAltitudeController:
    """A small state machine separating sensing scale from motion prediction."""

    def __init__(self, config: ControllerConfig = ControllerConfig()):
        self.config = config

    def initial_state(self) -> ControllerState:
        return ControllerState(current_level=len(self.config.altitude_levels_m) - 1)

    def step(self, state: ControllerState, signals: ControllerSignals) -> ControllerDecision:
        entropy, margin = _distribution_diagnostics(signals.goal_probabilities)
        state.goal_history.append(tuple(map(float, signals.predicted_goal_xy)))
        state.goal_history[:] = state.goal_history[-self.config.stable_goal_window:]
        stable = _goal_is_stable(state.goal_history, self.config.stable_goal_radius)

        confidence = (
            0.42 * (1.0 - entropy)
            + 0.18 * np.clip(margin, 0.0, 1.0)
            + 0.18 * float(stable)
            + 0.12 * np.clip(signals.landmark_confidence, 0.0, 1.0)
            + 0.10 * np.clip(signals.semantic_confidence, 0.0, 1.0)
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        recovery = entropy >= self.config.recovery_entropy
        state.recovery_steps = state.recovery_steps + 1 if recovery else 0
        if (self.config.allow_recovery_ascent
                and state.recovery_steps >= self.config.recovery_confirm_steps
                and state.phase > NavigationPhase.SEARCH):
            state.phase = NavigationPhase(state.phase - 1)
            state.phase_evidence_steps = 0
            state.recovery_steps = 0
        elif state.phase < NavigationPhase.REFINEMENT:
            next_phase = NavigationPhase(state.phase + 1)
            enough_confidence = confidence >= self.config.phase_enter_confidence[next_phase]
            if next_phase == NavigationPhase.REFINEMENT:
                enough_confidence &= signals.predicted_distance_m <= self.config.refinement_distance_m
                enough_confidence &= stable
            state.phase_evidence_steps = state.phase_evidence_steps + 1 if enough_confidence else 0
            if state.phase_evidence_steps >= self.config.transition_confirm_steps:
                state.phase = next_phase
                state.phase_evidence_steps = 0

        requested_level = self.config.phase_target_level[state.phase]
        if requested_level > state.current_level:
            action = AltitudeAction.ASCEND
            state.current_level += 1
        elif requested_level < state.current_level:
            action = AltitudeAction.DESCEND
            state.current_level -= 1
        else:
            action = AltitudeAction.HOLD

        # Stop control is intentionally a standalone conjunction.  Diagnostics
        # must never be inserted as an elif before this condition.
        stop_evidence = (
            state.phase == NavigationPhase.REFINEMENT
            and signals.predicted_distance_m <= self.config.stop_distance_m
            and signals.stop_probability >= self.config.stop_probability
            and stable
        )
        state.stop_evidence_steps = state.stop_evidence_steps + 1 if stop_evidence else 0
        should_stop = state.stop_evidence_steps >= self.config.stop_confirm_steps

        return ControllerDecision(
            phase=state.phase,
            altitude_action=action,
            target_level=state.current_level,
            target_altitude_m=self.config.altitude_levels_m[state.current_level],
            horizontal_step_m=self.config.phase_step_m[state.phase],
            should_stop=should_stop,
            confidence=confidence,
            normalized_entropy=entropy,
            top2_margin=margin,
            stable_goal=stable,
            recovery=recovery,
        )
