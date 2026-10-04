"""Offline CityNav samples for static-belief training.

This module intentionally does not import the environment, RGB crop client,
controller, or trajectory rollout code.  Every sample is fixed at episode start.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from multiagent.cityreferobject import (
    filter_landmarks,
    get_city_refer_objects,
    remove_duplicate_landmarks_by_area,
)
from multiagent.mapdata import MAP_BOUNDS


VARIANTS = ("instruction", "global", "referenced", "start_pose")


@dataclass(frozen=True)
class StaticBeliefSample:
    instruction: str
    map_name: str
    referenced_landmarks: tuple[str, ...]
    goal_row_col: tuple[float, float]
    start_row_col: tuple[float, float]
    start_yaw: float


def _world_to_row_col(map_name: str, xy, map_size: int, map_meters: float):
    bounds = MAP_BOUNDS[map_name]
    pixels_per_meter = map_size / map_meters
    return (
        (bounds.y_max - float(xy[1])) * pixels_per_meter,
        (float(xy[0]) - bounds.x_min) * pixels_per_meter,
    )


def load_static_samples(data_root: Path, split: str, map_size=240, map_meters=410.0):
    if split == "test_unseen":
        raise ValueError("test_unseen is forbidden for static-belief development")
    data_root = Path(data_root)
    objects = get_city_refer_objects(
        data_root / "cityrefer/objects.json",
        data_root / "cityrefer/processed_descriptions.json",
    )
    with (data_root / "processed_citynav" / f"citynav_{split}.json").open() as stream:
        trajectories = json.load(stream)
    samples = []
    for row in trajectories:
        map_name = f"{row['area']}_block_{row['block']}"
        object_id, description_id = int(row["object_ids"][0]), int(row["ann_ids"][0])
        target_object = objects[map_name][object_id]
        if description_id >= len(target_object.processed_descriptions):
            continue
        processed = target_object.processed_descriptions[description_id]
        pose = row["trajectory"][0]
        goal = row["target_positions"][-1]
        samples.append(StaticBeliefSample(
            instruction=target_object.descriptions[description_id],
            map_name=map_name,
            referenced_landmarks=tuple(processed.landmarks),
            goal_row_col=_world_to_row_col(map_name, goal, map_size, map_meters),
            start_row_col=_world_to_row_col(map_name, pose, map_size, map_meters),
            start_yaw=float(pose[3]) if len(pose) == 5 else float(np.arctan2(pose[4], pose[3])),
        ))
    return samples


class StaticMapRasterizer:
    """Cache global and individual named-landmark masks as compact uint8 arrays."""

    def __init__(self, data_root: Path, map_size=240, map_meters=410.0):
        data_root = Path(data_root)
        objects = get_city_refer_objects(
            data_root / "cityrefer/objects.json",
            data_root / "cityrefer/processed_descriptions.json",
        )
        self.landmarks = remove_duplicate_landmarks_by_area(filter_landmarks(objects))
        self.map_size = int(map_size)
        self.pixels_per_meter = self.map_size / float(map_meters)
        self._global = {}
        self._single = {}

    def _contour_pixels(self, map_name, landmark):
        bounds = MAP_BOUNDS[map_name]
        values = []
        for point in landmark.contour:
            col = round((point.x - bounds.x_min) * self.pixels_per_meter)
            row = round((bounds.y_max - point.y) * self.pixels_per_meter)
            values.append((col, row))
        return np.asarray(values, np.int32)

    def _mask(self, map_name, landmarks):
        result = np.zeros((self.map_size, self.map_size), np.uint8)
        for landmark in landmarks:
            contour = self._contour_pixels(map_name, landmark)
            if len(contour) >= 3:
                cv2.fillPoly(result, [contour], 1)
            elif len(contour) >= 2:
                cv2.polylines(result, [contour], False, 1, thickness=1)
        return result

    def global_mask(self, map_name):
        if map_name not in self._global:
            self._global[map_name] = self._mask(map_name, self.landmarks[map_name].values())
        return self._global[map_name]

    def landmark_mask(self, map_name, name):
        key = map_name, name
        if key not in self._single:
            candidates = list(self.landmarks[map_name].values())
            exact = [landmark for landmark in candidates if landmark.name == name]
            if not exact:
                exact = [landmark for landmark in candidates if landmark.name.casefold() == name.casefold()]
            self._single[key] = self._mask(map_name, exact)
        return self._single[key]

    def referenced_mask(self, map_name, names):
        result = np.zeros((self.map_size, self.map_size), np.uint8)
        for name in names:
            result |= self.landmark_mask(map_name, name)
        return result


class StaticBeliefDataset(Dataset):
    """Return a fixed 5-channel tensor for fair A/B/C/D input ablations.

    Channels are global landmarks, referenced landmarks, start-position impulse,
    cos(start yaw), and sin(start yaw).  Disabled ablation inputs are zeroed while
    architecture and parameter count stay unchanged.
    """

    def __init__(self, samples, rasterizer, language_lookup, variant, map_size=240):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.samples = samples
        self.rasterizer = rasterizer
        self.language_lookup = language_lookup
        self.variant = variant
        self.map_size = int(map_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        value = np.zeros((5, self.map_size, self.map_size), np.float32)
        if self.variant in {"global", "referenced", "start_pose"}:
            value[0] = self.rasterizer.global_mask(sample.map_name)
        if self.variant in {"referenced", "start_pose"}:
            value[1] = self.rasterizer.referenced_mask(sample.map_name, sample.referenced_landmarks)
        if self.variant == "start_pose":
            row, col = sample.start_row_col
            row, col = int(np.clip(round(row), 0, self.map_size - 1)), int(np.clip(round(col), 0, self.map_size - 1))
            value[2, row, col] = 1.0
            value[3].fill(np.cos(sample.start_yaw))
            value[4].fill(np.sin(sample.start_yaw))
        return {
            "static_map": torch.from_numpy(value),
            "language": self.language_lookup[sample.instruction],
            "goal_row_col": torch.tensor(sample.goal_row_col, dtype=torch.float32),
        }
