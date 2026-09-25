"""Strict CityNav/HETT rollout primitives for independent navigation models.

The policy-facing code in this module never receives a reference target.  Goal
coordinates are attached to completed trajectories only after action generation
so that the official metric implementation can score them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np

from multiagent.space import Point2D, Point3D, Pose4D
from multiagent.teacher.algorithm.lookahead import lookahead_discrete_action
from multiagent.teacher.trajectory import _moved_pose


@dataclass(frozen=True)
class OfficialProtocol:
    feedback: str = "student"
    max_action_len: int = 20
    move_iteration: int = 10
    grid_size: int = 5
    altitude_m: float = 50.0
    success_distance_m: float = 20.0
    map_size_m: float = 410.0
    low_level_stride_m: float = 5.0

    def validate(self) -> None:
        expected = {
            "feedback": "student",
            "max_action_len": 20,
            "move_iteration": 10,
            "grid_size": 5,
            "altitude_m": 50.0,
            "success_distance_m": 20.0,
        }
        actual = {name: getattr(self, name) for name in expected}
        wrong = {name: (actual[name], value) for name, value in expected.items()
                 if actual[name] != value}
        if wrong:
            raise ValueError(f"not the frozen HETT protocol: {wrong}")

    @property
    def maximum_translation_per_action_m(self) -> float:
        return self.move_iteration * self.low_level_stride_m


def canonical_instruction(text: str) -> str:
    """Canonical identity check for processed/raw copies of one instruction.

    The candidate builder reads ``processed_descriptions.json`` whereas the
    official environment reads the raw CityNav trajectory text.  Their known
    edits are whitespace and punctuation (for example ``L`` versus ``"L"``).
    Keeping only case-folded alphanumerics verifies content and order without
    consulting the target coordinate.
    """
    return "".join(character.casefold() for character in text
                   if character.isalnum())


def belief_argmax_xy(logits) -> tuple[float, float]:
    """Convert a ``[H,W]`` belief map argmax to normalized map ``(x,y)``."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("belief logits must be a square 2-D map")
    size = int(logits.shape[0])
    row, column = divmod(int(logits.argmax()), size)
    return (column + 0.5) / size, (row + 0.5) / size


def belief_distinct_topk_xy(logits, k: int = 2,
                            minimum_separation_m: float = 25.0,
                            map_size_m: float = 410.0):
    """Return spatially distinct belief modes instead of adjacent grid cells."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("belief logits must be a square 2-D map")
    size = int(logits.shape[0])
    selected = []
    for index in logits.flatten().argsort(descending=True).tolist():
        row, column = divmod(index, size)
        point = ((column + 0.5) / size, (row + 0.5) / size)
        if all(np.linalg.norm(np.asarray(point) - np.asarray(previous))
               * map_size_m > minimum_separation_m for previous in selected):
            selected.append(point)
            if len(selected) == k:
                break
    return selected


def optimistic_translation_steps(start_xy, destination_xy,
                                 protocol: OfficialProtocol) -> int:
    """Lower bound on official outer actions needed for a translation.

    This deliberately ignores turning, so it is safe only as a feasibility
    filter: an action rejected by this lower bound cannot fit the budget.
    """
    distance_m = float(np.linalg.norm(
        np.asarray(destination_xy, dtype=np.float64)
        - np.asarray(start_xy, dtype=np.float64))) * protocol.map_size_m
    return int(ceil(distance_m / protocol.maximum_translation_per_action_m))


def route_fits_budget(current_xy, via_xy, goal_xy, remaining_actions: int,
                      protocol: OfficialProtocol) -> bool:
    needed = (optimistic_translation_steps(current_xy, via_xy, protocol)
              + optimistic_translation_steps(via_xy, goal_xy, protocol))
    return needed <= remaining_actions


def move_toward(pose: Pose4D, destination: Point2D,
                move_iteration: int) -> Pose4D:
    """Use the exact discrete lookahead movement used by HETT stage one."""
    destination_3d = Point3D(destination.x, destination.y, pose.z)
    for _ in range(move_iteration):
        action = lookahead_discrete_action(pose, [destination_3d])
        pose = _moved_pose(pose, *action.value)
    return pose


@dataclass
class RolloutTrace:
    trajectory: list[Pose4D]
    action_count: int = 0
    movement_actions: int = 0
    stop_step: int | None = None
    online_observations: int = 1


@dataclass(frozen=True)
class BudgetedWaypoint:
    """One official-action waypoint, not an unconstrained camera pose."""
    name: str
    xy: tuple[float, float]


def _one_action_waypoint(current_xy, desired_xy, protocol):
    current = np.asarray(current_xy, dtype=np.float64)
    delta = np.asarray(desired_xy, dtype=np.float64) - current
    length = float(np.linalg.norm(delta))
    maximum = protocol.maximum_translation_per_action_m / protocol.map_size_m
    if length > maximum:
        delta *= maximum / length
    return tuple(np.clip(current + delta, 0.0, 1.0))


def generate_budgeted_waypoints(current_xy, top1_xy, top2_xy,
                                remaining_actions: int,
                                protocol: OfficialProtocol,
                                lateral_offset_m: float = 10.0):
    """Generate path-local candidates for the future joint action head.

    Every waypoint is reachable within one official outer action and must leave
    an optimistic route to the current Top-1 within the remaining budget.  The
    lateral options are small path offsets, never orbit viewpoints.
    """
    current = np.asarray(current_xy, dtype=np.float64)
    top1 = np.asarray(top1_xy, dtype=np.float64)
    top2 = np.asarray(top2_xy, dtype=np.float64)
    midpoint = (top1 + top2) / 2.0
    desired = {
        "toward_top1": top1,
        "toward_top2": top2,
        "toward_midpoint": midpoint,
    }
    direction = midpoint - current
    norm = float(np.linalg.norm(direction))
    if norm > 1e-8:
        progress = np.asarray(_one_action_waypoint(current, midpoint, protocol))
        perpendicular = np.array([-direction[1], direction[0]]) / norm
        offset = lateral_offset_m / protocol.map_size_m
        desired["path_left"] = progress + perpendicular * offset
        desired["path_right"] = progress - perpendicular * offset

    candidates = []
    for name, point in desired.items():
        waypoint = _one_action_waypoint(current, point, protocol)
        if route_fits_budget(
                current, waypoint, top1, remaining_actions, protocol):
            candidates.append(BudgetedWaypoint(name, waypoint))
    return candidates


def rollout_static_goals(env, predicted_goal_xy: Sequence[tuple[float, float]],
                         protocol: OfficialProtocol,
                         predicted_stop_radius_m: float = 5.0):
    """Roll out static predicted goals in the official online environment.

    RGB is freshly obtained by ``env._get_obs`` before every decision.  This
    stage-C policy intentionally uses pose/map identity but not RGB pixels; RGB
    evidence is the next controlled increment.  No target coordinate is an
    argument to this function.
    """
    protocol.validate()
    observations = env._get_obs(random_direction=False)
    batch_size = len(observations)
    if len(predicted_goal_xy) != batch_size:
        raise ValueError("one predicted goal is required per environment item")
    poses = [item["pose"] for item in observations]
    traces = [RolloutTrace([pose]) for pose in poses]
    ended = np.zeros(batch_size, dtype=bool)

    for outer_step in range(protocol.max_action_len):
        moved_any = False
        for index, observation in enumerate(observations):
            if ended[index]:
                continue
            trace = traces[index]
            trace.action_count += 1
            destination = env.unnormalize_position(
                predicted_goal_xy[index], observation["map_name"],
                protocol.map_size_m)
            if poses[index].xy.dist_to(destination) <= predicted_stop_radius_m:
                ended[index] = True
                trace.stop_step = outer_step + 1
                continue
            poses[index] = move_toward(
                poses[index], destination, protocol.move_iteration)
            trace.trajectory.append(poses[index])
            trace.movement_actions += 1
            moved_any = True

        if ended.all() or outer_step + 1 == protocol.max_action_len:
            break
        # This is the online-observation boundary.  The next action is based on
        # the new pose returned in this observation, never a precomputed route.
        observations = env._get_obs(poses, random_direction=False)
        for index, is_ended in enumerate(ended):
            if not is_ended:
                traces[index].online_observations += 1
        if not moved_any:
            break
    return traces


def attach_metric_only_ground_truth(env, traces: Sequence[RolloutTrace]):
    """Build official metric records after all policy decisions are complete."""
    if len(traces) != len(env.batch):
        raise ValueError("trace batch does not align with environment batch")
    results = []
    for episode, trace in zip(env.batch, traces):
        results.append({
            "instr_id": episode.id,
            "trajectory": trace.trajectory,
            "stage1_trajectory": trace.trajectory,
            "stage2_trajectory": [],
            "gt_trajectory": episode.trajectory,
            "goal": episode.target_position.xy,
            "camera_altitudes_m": [50.0] * len(trace.trajectory),
        })
    return results
