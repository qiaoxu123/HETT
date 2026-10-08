"""GT-free bounded execution of a predicted heatmap waypoint."""
import math
import numpy as np
from multiagent.space import Pose4D


def bounded_heatmap_step(pose, waypoint_xy, max_step_m=50.0):
    """Move at most max_step_m horizontally, retaining altitude.

    The waypoint is a model prediction. There is no target annotation,
    confidence threshold, learned direction head or learned progress input.
    """
    if not math.isfinite(max_step_m) or max_step_m <= 0:
        raise ValueError('max_step_m must be finite and positive')
    delta = np.asarray(waypoint_xy, dtype=np.float64) - np.asarray(pose.xy)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError('waypoint must have two finite coordinates')
    length = float(np.linalg.norm(delta))
    step = delta * min(1.0, max_step_m / max(length, 1e-9))
    yaw = float(np.arctan2(delta[1], delta[0])) if length > 1e-9 else pose.yaw
    return Pose4D(pose.x + step[0], pose.y + step[1], pose.z, yaw)
