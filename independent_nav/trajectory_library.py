"""Target-free rule trajectory hypotheses used before diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np
import torch

from independent_nav.official_protocol import OfficialProtocol


@dataclass(frozen=True)
class TrajectoryHypothesis:
    name: str
    actions: torch.Tensor       # [T,4]: dx, dy, sin(delta_yaw), cos(delta_yaw)
    valid: torch.Tensor         # [T]
    positions: torch.Tensor     # [T+1,2], padded with final position
    movement_steps: int
    path_length_m: float
    reaches_final_waypoint: bool


def wrap_angle(value):
    return (float(value) + pi) % (2 * pi) - pi


def encode_waypoint_path(start_xy, start_yaw, waypoints, horizon=20,
                         protocol=None, name="path"):
    """Convert normalized map waypoints to bounded official-scale actions."""
    protocol = protocol or OfficialProtocol()
    maximum = (protocol.maximum_translation_per_action_m
               / protocol.map_size_m)
    current = np.asarray(start_xy, dtype=np.float64)
    heading = float(start_yaw)
    actions = torch.zeros(horizon, 4)
    valid = torch.zeros(horizon, dtype=torch.bool)
    positions = [current.copy()]
    path_length = 0.0
    step_index = 0
    reaches = True
    for waypoint in waypoints:
        destination = np.clip(np.asarray(waypoint, dtype=np.float64), 0, 1)
        while np.linalg.norm(destination - current) > 1e-8:
            if step_index >= horizon:
                reaches = False
                break
            delta = destination - current
            length = float(np.linalg.norm(delta))
            step = delta if length <= maximum else delta * (maximum / length)
            # Normalized map y grows down, while world yaw uses y growing up.
            next_heading = atan2(-float(step[1]), float(step[0]))
            delta_yaw = wrap_angle(next_heading - heading)
            actions[step_index] = torch.tensor((
                float(step[0] / maximum), float(step[1] / maximum),
                sin(delta_yaw), cos(delta_yaw)))
            valid[step_index] = True
            current = np.clip(current + step, 0, 1)
            positions.append(current.copy())
            path_length += float(np.linalg.norm(step) * protocol.map_size_m)
            heading = next_heading
            step_index += 1
        if not reaches:
            break
    while len(positions) < horizon + 1:
        positions.append(current.copy())
    return TrajectoryHypothesis(
        name=name, actions=actions, valid=valid,
        positions=torch.tensor(np.stack(positions), dtype=torch.float32),
        movement_steps=step_index, path_length_m=path_length,
        reaches_final_waypoint=reaches)


def generate_rule_trajectory_set(start_xy, start_yaw, belief_modes,
                                 horizon=20, lateral_offset_m=10.0,
                                 protocol=None):
    """Generate eight target-free navigation-and-observation hypotheses."""
    protocol = protocol or OfficialProtocol()
    modes = [np.asarray(point, dtype=np.float64) for point in belief_modes]
    if len(modes) < 5:
        raise ValueError("five distinct belief modes are required")
    output = [encode_waypoint_path(
        start_xy, start_yaw, [modes[index]], horizon, protocol,
        f"direct_top{index + 1}") for index in range(5)]

    first, second = modes[:2]
    midpoint = (first + second) / 2.0
    direction = midpoint - np.asarray(start_xy, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    perpendicular = (np.asarray([-direction[1], direction[0]]) / norm
                     if norm > 1e-8 else np.asarray([0.0, 1.0]))
    offset = perpendicular * (lateral_offset_m / protocol.map_size_m)
    output.extend((
        encode_waypoint_path(
            start_xy, start_yaw, [midpoint, first], horizon, protocol,
            "midpoint_disambiguate_then_top1"),
        encode_waypoint_path(
            start_xy, start_yaw, [midpoint + offset, first], horizon, protocol,
            "left_observe_then_top1"),
        encode_waypoint_path(
            start_xy, start_yaw, [midpoint - offset, second], horizon, protocol,
            "right_observe_then_top2"),
    ))
    return output


def stack_trajectory_set(values):
    return {
        "actions": torch.stack([value.actions for value in values]),
        "valid": torch.stack([value.valid for value in values]),
        "positions": torch.stack([value.positions for value in values]),
        "movement_steps": torch.tensor(
            [value.movement_steps for value in values]),
        "path_length_m": torch.tensor(
            [value.path_length_m for value in values]),
        "reaches_final_waypoint": torch.tensor(
            [value.reaches_final_waypoint for value in values]),
        "names": [value.name for value in values],
    }

