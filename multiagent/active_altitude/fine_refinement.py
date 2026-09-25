"""Fine-stage-only altitude refinement for the fixed-50 m HETT pipeline.

The global navigation map remains at a canonical altitude.  This controller
only changes the RGB camera altitude after the coarse goal is close, stable and
contained by the next lower field of view.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from math import log
from typing import Sequence, Tuple

import numpy as np

from multiagent.space import Pose4D

from .controller import AltitudeAction


class FineNavigationPhase(IntEnum):
    COARSE = 0
    REFINE = 1


@dataclass(frozen=True)
class FineAltitudeConfig:
    """Configuration for a 50 -> 40 -> 30 m visual-probe schedule."""

    altitude_levels_m: Tuple[float, ...] = (50.0, 40.0, 30.0)
    enter_distance_m: float = 40.0
    exit_distance_m: float = 55.0
    containment_margin_m: float = 5.0
    enter_confidence: float = 0.55
    descend_confidence: float = 0.62
    descend_max_entropy: float = 0.62
    recovery_entropy: float = 0.82
    stable_goal_radius: float = 0.035
    stable_goal_window: int = 3
    transition_confirm_steps: int = 2
    recovery_confirm_steps: int = 2

    def __post_init__(self):
        if len(self.altitude_levels_m) < 2:
            raise ValueError("fine altitude refinement needs at least two levels")
        if any(lower >= upper for upper, lower in zip(
                self.altitude_levels_m, self.altitude_levels_m[1:])):
            raise ValueError("fine altitude levels must be strictly high-to-low")
        if self.containment_margin_m < 0:
            raise ValueError("containment margin must be non-negative")
        if self.stable_goal_window < 2:
            raise ValueError("stable goal window must contain at least two observations")


@dataclass(frozen=True)
class FineAltitudeSignals:
    goal_probabilities: Sequence[float]
    predicted_goal_xy: Tuple[float, float]
    predicted_distance_m: float


@dataclass
class FineAltitudeState:
    phase: FineNavigationPhase = FineNavigationPhase.COARSE
    current_level: int = 0
    transition_steps: int = 0
    recovery_steps: int = 0
    goal_history: list[Tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class FineAltitudeDecision:
    phase: FineNavigationPhase
    altitude_action: AltitudeAction
    target_altitude_m: float
    confidence: float
    normalized_entropy: float
    top2_margin: float
    stable_goal: bool
    contained_by_next_view: bool
    global_memory_eligible: bool
    recovery: bool


def canonical_map_pose(pose: Pose4D, ground_level: float,
                       canonical_altitude_m: float = 50.0) -> Pose4D:
    """Keep XY/yaw while projecting map coverage to the canonical altitude."""
    return Pose4D(pose.x, pose.y, ground_level + canonical_altitude_m, pose.yaw)


def _diagnostics(probabilities: Sequence[float]) -> tuple[float, float]:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size < 2 or not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("goal probabilities must be finite and non-negative")
    total = probs.sum()
    if total <= 0:
        raise ValueError("goal probabilities must have positive mass")
    probs /= total
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None)))) / log(probs.size)
    top2 = np.partition(probs, -2)[-2:]
    return float(np.clip(entropy, 0.0, 1.0)), float(top2.max() - top2.min())


def _stable(history: Sequence[Tuple[float, float]], window: int, radius: float) -> bool:
    if len(history) < window:
        return False
    points = np.asarray(history[-window:], dtype=np.float64)
    center = points.mean(axis=0)
    return bool(np.linalg.norm(points - center, axis=1).max() <= radius)


class FineAltitudeController:
    """Choose camera height without changing coarse motion or stop decisions."""

    def __init__(self, config: FineAltitudeConfig = FineAltitudeConfig()):
        self.config = config

    def initial_state(self) -> FineAltitudeState:
        return FineAltitudeState()

    def step(self, state: FineAltitudeState,
             signals: FineAltitudeSignals) -> FineAltitudeDecision:
        entropy, margin = _diagnostics(signals.goal_probabilities)
        state.goal_history.append(tuple(map(float, signals.predicted_goal_xy)))
        state.goal_history[:] = state.goal_history[-self.config.stable_goal_window:]
        stable = _stable(
            state.goal_history,
            self.config.stable_goal_window,
            self.config.stable_goal_radius,
        )
        confidence = float(np.clip(
            0.55 * (1.0 - entropy) + 0.25 * margin + 0.20 * float(stable),
            0.0,
            1.0,
        ))

        next_level = min(state.current_level + 1, len(self.config.altitude_levels_m) - 1)
        next_altitude = self.config.altitude_levels_m[next_level]
        contained = (
            signals.predicted_distance_m
            <= next_altitude - self.config.containment_margin_m
        )
        action = AltitudeAction.HOLD
        recovery = False

        if state.phase == FineNavigationPhase.COARSE:
            enter = (
                stable
                and signals.predicted_distance_m <= self.config.enter_distance_m
                and contained
                and confidence >= self.config.enter_confidence
            )
            state.transition_steps = state.transition_steps + 1 if enter else 0
            if state.transition_steps >= self.config.transition_confirm_steps:
                state.phase = FineNavigationPhase.REFINE
                state.current_level = 1
                state.transition_steps = 0
                action = AltitudeAction.DESCEND
        else:
            recovery = (
                entropy >= self.config.recovery_entropy
                or (len(state.goal_history) >= self.config.stable_goal_window and not stable)
                or signals.predicted_distance_m > self.config.exit_distance_m
            )
            state.recovery_steps = state.recovery_steps + 1 if recovery else 0
            if state.recovery_steps >= self.config.recovery_confirm_steps:
                if state.current_level > 0:
                    state.current_level -= 1
                    action = AltitudeAction.ASCEND
                if state.current_level == 0:
                    state.phase = FineNavigationPhase.COARSE
                state.recovery_steps = 0
                state.transition_steps = 0
            else:
                descend = (
                    state.current_level < len(self.config.altitude_levels_m) - 1
                    and stable
                    and contained
                    and entropy <= self.config.descend_max_entropy
                    and confidence >= self.config.descend_confidence
                )
                state.transition_steps = state.transition_steps + 1 if descend else 0
                if state.transition_steps >= self.config.transition_confirm_steps:
                    state.current_level += 1
                    state.transition_steps = 0
                    action = AltitudeAction.DESCEND

        target_altitude = self.config.altitude_levels_m[state.current_level]
        return FineAltitudeDecision(
            phase=state.phase,
            altitude_action=action,
            target_altitude_m=target_altitude,
            confidence=confidence,
            normalized_entropy=entropy,
            top2_margin=margin,
            stable_goal=stable,
            contained_by_next_view=contained,
            global_memory_eligible=state.current_level == 0,
            recovery=recovery,
        )


def oracle_training_altitude(stage1_ended: bool, distance_to_target_m: float,
                             config: FineAltitudeConfig) -> float:
    """Teacher-forced camera altitude used only while optimizing HETT.

    The privileged target distance is deliberately isolated in this helper so
    evaluation cannot accidentally use it.  Coarse navigation always stays at
    the canonical first level; fine navigation uses the lowest configured
    level only after the true target is inside the fine-stage radius.
    """
    if stage1_ended and distance_to_target_m <= config.enter_distance_m:
        return config.altitude_levels_m[-1]
    return config.altitude_levels_m[0]
