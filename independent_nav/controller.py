"""A minimal HETT-independent geometric action controller.

The controller consumes a world-coordinate goal and emits the same conceptual
actions as the simulator (30 degree turns, 5 metre forward moves, and stop),
but it has no dependency on HETT models, agents, or candidate points.
"""

from dataclasses import dataclass
from math import atan2, cos, pi, sin
from typing import List, Tuple


STOP = "stop"
MOVE_FORWARD = "move_forward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"


def wrap_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class ControllerConfig:
    stride_m: float = 5.0
    turn_radians: float = pi / 6.0
    heading_tolerance_radians: float = pi / 12.0
    stop_radius_m: float = 5.0
    max_actions: int = 200


@dataclass(frozen=True)
class PlanResult:
    actions: List[str]
    path_xy: List[Tuple[float, float]]
    final_yaw: float


class GeometricGoalController:
    """Turn toward a predicted goal, fly forward, then stop near that goal."""

    def __init__(self, config: ControllerConfig = ControllerConfig()):
        self.config = config

    def plan(
        self,
        start_xy: Tuple[float, float],
        start_yaw: float,
        goal_xy: Tuple[float, float],
    ) -> PlanResult:
        x, y = map(float, start_xy)
        gx, gy = map(float, goal_xy)
        yaw = float(start_yaw)
        path = [(x, y)]
        actions: List[str] = []

        for _ in range(self.config.max_actions):
            dx, dy = gx - x, gy - y
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= self.config.stop_radius_m:
                actions.append(STOP)
                break

            desired = atan2(dy, dx)
            error = wrap_angle(desired - yaw)
            if error > self.config.heading_tolerance_radians:
                yaw = wrap_angle(yaw + self.config.turn_radians)
                actions.append(TURN_LEFT)
                continue
            if error < -self.config.heading_tolerance_radians:
                yaw = wrap_angle(yaw - self.config.turn_radians)
                actions.append(TURN_RIGHT)
                continue

            x += self.config.stride_m * cos(yaw)
            y += self.config.stride_m * sin(yaw)
            path.append((x, y))
            actions.append(MOVE_FORWARD)
        else:
            actions.append(STOP)

        return PlanResult(actions=actions, path_xy=path, final_yaw=yaw)
