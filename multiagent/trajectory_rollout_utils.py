"""Minimal rollout book-keeping helpers (no simulator/model imports)."""

import math


def should_record_pose(previous_pose, new_pose, *, ended):
    """Keep a newly executed terminal action in the scored trajectory.

    A stop without movement should not append a synthetic extra step.
    In-progress episodes retain legacy per-timestep recording semantics.
    """
    return not bool(ended) or new_pose != previous_pose


def wrapped_yaw_delta_rad(new_yaw, old_yaw):
    """Return the absolute shortest angular difference in radians."""
    return abs(math.atan2(math.sin(new_yaw - old_yaw),
                          math.cos(new_yaw - old_yaw)))
