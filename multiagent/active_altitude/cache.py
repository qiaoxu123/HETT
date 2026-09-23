"""Bounded on-demand observation cache; avoids exhaustive pose rendering."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import numpy as np


T = TypeVar("T")


@dataclass(frozen=True)
class ObservationKey:
    map_name: str
    x_cell: int
    y_cell: int
    yaw_bin: int
    altitude_level: int


def quantize_observation(map_name: str, x_m: float, y_m: float, yaw_rad: float,
                         altitude_level: int, xy_resolution_m: float = 5.0,
                         yaw_bins: int = 16) -> ObservationKey:
    if xy_resolution_m <= 0 or yaw_bins < 1:
        raise ValueError("cache quantization must be positive")
    yaw = float(yaw_rad) % (2 * np.pi)
    return ObservationKey(
        map_name=map_name,
        x_cell=int(np.floor(x_m / xy_resolution_m)),
        y_cell=int(np.floor(y_m / xy_resolution_m)),
        yaw_bin=int(np.floor(yaw / (2 * np.pi) * yaw_bins)) % yaw_bins,
        altitude_level=int(altitude_level),
    )


class ObservationLRU(Generic[T]):
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._values: OrderedDict[ObservationKey, T] = OrderedDict()

    def get(self, key: ObservationKey) -> Optional[T]:
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key: ObservationKey, value: T):
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        return len(self._values)
