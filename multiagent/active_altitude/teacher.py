"""Privileged training targets kept separate from deployable control."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .controller import AltitudeAction, ControllerConfig, NavigationPhase


@dataclass(frozen=True)
class TeacherTargets:
    phase: NavigationPhase
    altitude_level_distribution: np.ndarray
    altitude_action_distribution: np.ndarray
    altitude_action: AltitudeAction
    stop_target: float
    human_agreement_weight: float


def _nearest_level(altitude_m: float, levels: Sequence[float]) -> int:
    return int(np.abs(np.asarray(levels, dtype=np.float64) - altitude_m).argmin())


def _oracle_phase(distance_to_target_m: float) -> NavigationPhase:
    if distance_to_target_m > 100.0:
        return NavigationPhase.SEARCH
    if distance_to_target_m > 60.0:
        return NavigationPhase.COARSE_APPROACH
    if distance_to_target_m > 20.0:
        return NavigationPhase.LANDMARK_APPROACH
    return NavigationPhase.REFINEMENT


def build_teacher_targets(distance_to_target_m: float, current_level: int,
                          human_altitude_m: Optional[float],
                          config: ControllerConfig = ControllerConfig(),
                          human_weight: float = 0.35) -> TeacherTargets:
    """Blend oracle phase structure with noisy human altitude as soft labels."""
    phase = _oracle_phase(float(distance_to_target_m))
    oracle_level = config.phase_target_level[phase]
    distribution = np.zeros(len(config.altitude_levels_m), dtype=np.float32)
    distribution[oracle_level] = 1.0
    agreement = 0.0
    if human_altitude_m is not None and np.isfinite(human_altitude_m):
        human_level = _nearest_level(float(human_altitude_m), config.altitude_levels_m)
        level_gap = abs(human_level - oracle_level)
        agreement = float(np.exp(-level_gap))
        weight = float(np.clip(human_weight * agreement, 0.0, 0.5))
        distribution *= 1.0 - weight
        distribution[human_level] += weight

    action_distribution = np.zeros(len(AltitudeAction), dtype=np.float32)
    for target_level, probability in enumerate(distribution):
        if target_level > current_level:
            action_distribution[AltitudeAction.ASCEND] += probability
        elif target_level < current_level:
            action_distribution[AltitudeAction.DESCEND] += probability
        else:
            action_distribution[AltitudeAction.HOLD] += probability
    action = AltitudeAction(int(action_distribution.argmax()))
    return TeacherTargets(
        phase=phase,
        altitude_level_distribution=distribution,
        altitude_action_distribution=action_distribution,
        altitude_action=action,
        stop_target=float(distance_to_target_m <= config.stop_distance_m),
        human_agreement_weight=agreement,
    )
