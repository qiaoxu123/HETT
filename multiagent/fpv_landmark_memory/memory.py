from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def observation_quality(visible_ratio: float, pixel_area: float, distance_m: float,
                        view_angle_deg: float) -> float:
    """Fixed, training-free quality heuristic; no split-specific fitting."""
    visible = min(max(float(visible_ratio), 0.0), 1.0)
    area = max(float(pixel_area), 0.0)
    distance = max(float(distance_m), 0.0)
    angle = min(abs(float(view_angle_deg)), 90.0)
    # Favor visible, large, nearby observations and mildly penalize grazing views.
    angle_quality = max(0.15, 1.0 - angle / 110.0)
    return visible * (min(area, 4096.0) / 4096.0) ** 0.5 * angle_quality / (1.0 + distance / 100.0)


@dataclass
class Observation:
    feature: Any
    quality: float
    visible_ratio: float
    distance_m: float
    view_angle_deg: float
    step: int


@dataclass
class LandmarkState:
    landmark_id: str
    map_position: tuple[float, float] | None = None
    category_or_name: str = ""
    observations: list[Observation] = field(default_factory=list)
    first_seen_step: int | None = None
    last_seen_step: int | None = None

    @property
    def best(self) -> Observation | None:
        return max(self.observations, key=lambda o: o.quality, default=None)

    @property
    def confidence(self) -> float:
        return self.best.quality if self.best else 0.0


class AppearanceMemory:
    """Bounded landmark-keyed buffer. Features are detached data, not a model."""

    def __init__(self, max_views_per_landmark: int = 5):
        if max_views_per_landmark < 1:
            raise ValueError("max_views_per_landmark must be >= 1")
        self.max_views = int(max_views_per_landmark)
        self.landmarks: dict[str, LandmarkState] = {}

    def update(self, landmark_id: str, feature: Any, *, step: int,
               visible_ratio: float, pixel_area: float, distance_m: float,
               view_angle_deg: float, map_position=None, category_or_name="") -> LandmarkState:
        key = str(landmark_id)
        state = self.landmarks.setdefault(key, LandmarkState(key))
        if map_position is not None:
            state.map_position = tuple(map(float, map_position[:2]))
        if category_or_name:
            state.category_or_name = str(category_or_name)
        quality = observation_quality(visible_ratio, pixel_area, distance_m, view_angle_deg)
        if visible_ratio > 0 and pixel_area > 0:
            state.observations.append(Observation(feature, quality, float(visible_ratio),
                                                  float(distance_m), float(view_angle_deg), int(step)))
            state.observations.sort(key=lambda x: (x.quality, -x.step), reverse=True)
            state.observations = state.observations[:self.max_views]
            state.first_seen_step = int(step) if state.first_seen_step is None else min(state.first_seen_step, int(step))
            state.last_seen_step = int(step) if state.last_seen_step is None else max(state.last_seen_step, int(step))
        return state

    def visible(self, landmark_id: str) -> bool:
        return bool(self.landmarks.get(str(landmark_id)) and self.landmarks[str(landmark_id)].observations)

    def get(self, landmark_id: str) -> LandmarkState | None:
        return self.landmarks.get(str(landmark_id))
