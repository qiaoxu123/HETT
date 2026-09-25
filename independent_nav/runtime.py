"""Target-free closed-loop runtime for the independent belief navigator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import atan2
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from independent_nav.belief import VisibilityGatedBeliefMap
from independent_nav.controller import GeometricGoalController


ASCEND = "ascend"
DESCEND = "descend"
OBSERVE = "observe"


class RuntimePhase(str, Enum):
    VERIFY = "verify"
    APPROACH = "approach"
    STOPPED = "stopped"


@dataclass(frozen=True)
class RuntimeConfig:
    map_size_m: float = 410.0
    success_radius_m: float = 20.0
    view_reached_radius_m: float = 5.0
    altitude_tolerance_m: float = 2.0
    maximum_observations: int = 2
    entropy_view_threshold: float = 0.72
    concentration_threshold: float = 0.60
    stop_type_confidence: float = 0.60
    stop_relation_confidence: float = 0.60
    stable_goal_radius_m: float = 10.0
    stable_goal_frames: int = 2
    stop_confirm_frames: int = 2
    approach_altitude_m: float = 50.0


@dataclass(frozen=True)
class ViewPlan:
    xy: Tuple[float, float]
    altitude_m: float
    yaw: float
    utility: float = 0.0


@dataclass(frozen=True)
class ObservationEvidence:
    log_likelihood_ratio: torch.Tensor
    observable: torch.Tensor
    quality: torch.Tensor | float
    evidence_id: str
    target_type_confidence: float
    relation_confidence: float


@dataclass
class RuntimeState:
    belief: VisibilityGatedBeliefMap
    phase: RuntimePhase = RuntimePhase.VERIFY
    observations: int = 0
    goal_history: list[Tuple[float, float]] = field(default_factory=list)
    stop_evidence: int = 0
    target_type_confidence: float = 0.0
    relation_confidence: float = 0.0
    active_view: Optional[ViewPlan] = None


@dataclass(frozen=True)
class RuntimeDecision:
    phase: RuntimePhase
    action: str
    predicted_goal_xy: Tuple[float, float]
    waypoint_xy: Tuple[float, float]
    target_altitude_m: float
    should_stop: bool
    normalized_entropy: float
    concentration_20m: float
    stable_goal: bool


ViewPlanner = Callable[[torch.Tensor, Tuple[float, float], float, int], ViewPlan]


class ActiveBeliefNavigator:
    """Alternate active observation, signed belief update, motion and stopping."""

    def __init__(self, prior_probability: torch.Tensor, view_planner: ViewPlanner,
                 config: RuntimeConfig = RuntimeConfig()):
        if prior_probability.ndim != 2:
            raise ValueError("prior belief must be a 2-D map")
        self.config = config
        self.state = RuntimeState(VisibilityGatedBeliefMap(prior_probability))
        self.view_planner = view_planner
        self.controller = GeometricGoalController()

    def _goal(self):
        probability = self.state.belief.normalized()
        height, width = probability.shape
        index = int(probability.argmax())
        row, column = divmod(index, width)
        return ((column + 0.5) / width, (row + 0.5) / height)

    def _diagnostics(self, goal):
        probability = self.state.belief.normalized()
        flat = probability.flatten()
        entropy = float(-(flat * flat.clamp_min(1e-8).log()).sum())
        entropy /= max(float(np.log(flat.numel())), 1e-8)
        height, width = probability.shape
        yy, xx = torch.meshgrid(
            (torch.arange(height) + 0.5) / height,
            (torch.arange(width) + 0.5) / width, indexing="ij")
        distance = torch.sqrt((xx - goal[0]) ** 2 + (yy - goal[1]) ** 2)
        local = distance * self.config.map_size_m <= self.config.success_radius_m
        concentration = float(probability[local].sum())
        history = self.state.goal_history[-self.config.stable_goal_frames:]
        stable = len(history) >= self.config.stable_goal_frames
        if stable:
            center = np.asarray(history).mean(0)
            stable = bool(max(np.linalg.norm(
                (np.asarray(point) - center) * self.config.map_size_m)
                for point in history) <= self.config.stable_goal_radius_m)
        return entropy, concentration, stable

    def _motion_action(self, current_xy, current_yaw, current_altitude, waypoint,
                       target_altitude):
        if current_altitude < target_altitude - self.config.altitude_tolerance_m:
            return ASCEND
        if current_altitude > target_altitude + self.config.altitude_tolerance_m:
            return DESCEND
        current_m = tuple(float(value) * self.config.map_size_m for value in current_xy)
        waypoint_m = tuple(float(value) * self.config.map_size_m for value in waypoint)
        if np.linalg.norm(np.asarray(current_m) - waypoint_m) <= self.config.view_reached_radius_m:
            return OBSERVE
        plan = self.controller.plan(current_m, current_yaw, waypoint_m)
        return plan.actions[0]

    def step(self, current_xy: Tuple[float, float], current_yaw: float,
             current_altitude_m: float,
             observation: Optional[ObservationEvidence] = None) -> RuntimeDecision:
        if observation is not None:
            before = self.state.belief.observations
            self.state.belief.update(
                observation.log_likelihood_ratio, observation.observable,
                observation.quality, observation.evidence_id)
            if self.state.belief.observations > before:
                self.state.observations += 1
                self.state.target_type_confidence = float(
                    observation.target_type_confidence)
                self.state.relation_confidence = float(
                    observation.relation_confidence)
                self.state.active_view = None

        goal = self._goal()
        self.state.goal_history.append(goal)
        self.state.goal_history[:] = self.state.goal_history[-self.config.stable_goal_frames:]
        entropy, concentration, stable = self._diagnostics(goal)
        current_distance = (np.linalg.norm(
            np.asarray(current_xy) - np.asarray(goal)) * self.config.map_size_m)
        stop_condition = (
            self.state.observations > 0
            and current_distance <= self.config.success_radius_m
            and concentration >= self.config.concentration_threshold
            and stable
            and self.state.target_type_confidence >= self.config.stop_type_confidence
            and self.state.relation_confidence >= self.config.stop_relation_confidence)
        self.state.stop_evidence = self.state.stop_evidence + 1 if stop_condition else 0
        should_stop = self.state.stop_evidence >= self.config.stop_confirm_frames
        if should_stop:
            self.state.phase = RuntimePhase.STOPPED
            return RuntimeDecision(
                self.state.phase, "stop", goal, goal, current_altitude_m, True,
                entropy, concentration, stable)

        needs_view = (
            self.state.observations < self.config.maximum_observations
            and (entropy >= self.config.entropy_view_threshold
                 or concentration < self.config.concentration_threshold))
        if needs_view:
            self.state.phase = RuntimePhase.VERIFY
            if self.state.active_view is None:
                self.state.active_view = self.view_planner(
                    self.state.belief.normalized(), current_xy,
                    current_altitude_m, self.state.observations)
            view = self.state.active_view
            action = self._motion_action(
                current_xy, current_yaw, current_altitude_m,
                view.xy, view.altitude_m)
            return RuntimeDecision(
                self.state.phase, action, goal, view.xy, view.altitude_m, False,
                entropy, concentration, stable)

        self.state.phase = RuntimePhase.APPROACH
        action = self._motion_action(
            current_xy, current_yaw, current_altitude_m,
            goal, self.config.approach_altitude_m)
        return RuntimeDecision(
            self.state.phase, action, goal, goal,
            self.config.approach_altitude_m, False,
            entropy, concentration, stable)
