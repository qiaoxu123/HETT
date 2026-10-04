"""Map-derived expected scene descriptors around a location.

These descriptors are computed from a fixed radius around a B0 peak.  They do
not select a nearest object and do not synthesize an RGB observation.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from multiagent.visual_attributes.labels import geometry_features, shape_label, size_label
from multiagent.visual_attributes.parser import parse_attributes


@lru_cache(maxsize=None)
def _description_attributes(texts):
    parsed = [parse_attributes(text) for text in texts]
    return {
        "color": tuple(sorted({value for item in parsed for value in item.get("color", ())})),
        "roof": tuple(sorted({value for item in parsed for value in item.get("roof", ())})),
    }


def expected_scene(xy, objects, *, radii=(20.0, 40.0, 60.0), size_thresholds=(100.0, 400.0)):
    center = np.asarray(xy, dtype=float)
    records = []
    for obj in objects:
        distance = float(np.linalg.norm(np.asarray(obj.position[:2], dtype=float) - center))
        if distance <= max(radii):
            geometry = geometry_features(obj)
            language = _description_attributes(tuple(obj.descriptions))
            records.append({
                "id": int(obj.id), "name": obj.name, "type": obj.object_type,
                "distance_m": distance, "bearing_rad": float(math.atan2(obj.position.y - center[1], obj.position.x - center[0])),
                "size": size_label(geometry["area_m2"], size_thresholds),
                "shape": shape_label(geometry), "area_m2": geometry["area_m2"],
                "color": list(language["color"]), "roof": list(language["roof"]),
            })
    counts = {}
    for radius in radii:
        selected = [item for item in records if item["distance_m"] <= radius]
        by_type = {}
        for item in selected:
            by_type[item["type"]] = by_type.get(item["type"], 0) + 1
        counts[str(int(radius))] = by_type
    return {"xy": [float(xy[0]), float(xy[1])], "radii_m": list(radii),
            "counts": counts, "objects": sorted(records, key=lambda item: item["distance_m"])}


def count_similarity(left: dict, right: dict, radius="40") -> float:
    a, b = left["counts"].get(str(radius), {}), right["counts"].get(str(radius), {})
    labels = set(a) | set(b)
    if not labels:
        return 1.0
    numerator = sum(abs(a.get(label, 0) - b.get(label, 0)) for label in labels)
    denominator = max(sum(a.values()) + sum(b.values()), 1)
    return float(1.0 - numerator / denominator)


def attribute_similarity(candidate: dict, target: dict) -> float:
    target_objects = target.get("objects", [])
    if not target_objects:
        return 0.0
    reference = target_objects[0]
    scores = []
    for item in candidate.get("objects", []):
        values = [item.get("type") == reference.get("type"),
                  item.get("size") == reference.get("size"),
                  item.get("shape") == reference.get("shape")]
        for name in ("color", "roof"):
            required = set(reference.get(name, ()))
            if required:
                values.append(bool(required.intersection(item.get(name, ()))))
        scores.append(sum(values) / len(values))
    return float(max(scores, default=0.0))


def landmark_geometry_similarity(candidate_xy, target_xy, landmarks) -> float:
    if not landmarks:
        return 0.0
    candidate, target = np.asarray(candidate_xy, dtype=float), np.asarray(target_xy, dtype=float)
    scores = []
    for landmark in landmarks:
        point = np.asarray(landmark.position[:2], dtype=float)
        candidate_distance, target_distance = np.linalg.norm(candidate - point), np.linalg.norm(target - point)
        distance_score = math.exp(-abs(candidate_distance - target_distance) / 20.0)
        candidate_bearing = math.atan2(point[1] - candidate[1], point[0] - candidate[0])
        target_bearing = math.atan2(point[1] - target[1], point[0] - target[0])
        angle = abs((candidate_bearing - target_bearing + math.pi) % (2 * math.pi) - math.pi)
        scores.append(0.5 * distance_score + 0.5 * (1.0 - angle / math.pi))
    return float(np.mean(scores))
