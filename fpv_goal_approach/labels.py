import math
from typing import Iterable, Sequence


DISTANCE_EDGES = (10.0, 20.0, 30.0, 40.0, 60.0)
DISTANCE_NAMES = ("<10m", "10-20m", "20-30m", "30-40m", "40-60m", ">60m")
BEARING_NAMES = ("far_left", "left", "front", "right", "far_right")
PHASE_NAMES = ("far", "approaching", "near_goal", "arrival")


def wrap_radians(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def distance_bin(distance_m: float) -> int:
    """Return the requested six-way bin, ordered from nearest to farthest."""
    for index, edge in enumerate(DISTANCE_EDGES):
        if distance_m < edge:
            return index
    return 5


def target_bearing(x: float, y: float, target_xy: Sequence[float]) -> float:
    return math.atan2(float(target_xy[1]) - y, float(target_xy[0]) - x)


def relative_bearing(x: float, y: float, yaw: float, target_xy: Sequence[float]) -> float:
    return wrap_radians(target_bearing(x, y, target_xy) - yaw)


def bearing_bin(relative_bearing_rad: float) -> int:
    """Five view-relative bins; negative angles are left of the camera."""
    degrees = math.degrees(wrap_radians(relative_bearing_rad))
    if degrees < -60.0:
        return 0
    if degrees < -20.0:
        return 1
    if degrees <= 20.0:
        return 2
    if degrees <= 60.0:
        return 3
    return 4


def phase(distance_m: float, success_dist: float = 20.0) -> int:
    if distance_m <= success_dist:
        return 3
    if distance_m <= 30.0:
        return 2
    if distance_m <= 60.0:
        return 1
    return 0


def geometric_arrival(distance_m: float, success_dist: float = 20.0) -> int:
    return int(distance_m <= success_dist)


def clip_indices(current_index: int, offsets: Iterable[int] = (0, 1, 3, 6)) -> list[int]:
    """Progressive history sampling, returned oldest to newest."""
    indices = [max(0, current_index - int(offset)) for offset in offsets]
    return list(reversed(indices))


def pose_is_duplicate(
    pose_a: Sequence[float], pose_b: Sequence[float], min_translation_m: float = 3.0,
    min_yaw_deg: float = 10.0,
) -> bool:
    translation = math.hypot(float(pose_a[0]) - float(pose_b[0]), float(pose_a[1]) - float(pose_b[1]))
    yaw_delta = abs(math.degrees(wrap_radians(float(pose_a[3]) - float(pose_b[3]))))
    return translation < min_translation_m and yaw_delta < min_yaw_deg

