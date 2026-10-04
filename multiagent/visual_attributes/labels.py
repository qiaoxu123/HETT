"""Geometry and weak-language labels for the attribute benchmark."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from shapely.geometry import Polygon

from .parser import parse_attributes


COLOR_CLASSES = ("white", "black", "gray", "red", "brown", "green", "blue", "yellow")
SIZE_CLASSES = ("small", "medium", "large")
SHAPE_CLASSES = ("square", "rectangular", "round", "irregular")
SEMANTIC_CLASSES = ("Building", "Car", "Ground", "Parking")
CONTEXT_CLASSES = ("road", "intersection", "parking_lot", "field", "grass", "water", "bridge", "railway", "vegetation")


def primary_language_label(texts, category, allowed):
    counts = Counter(value for text in texts for value in parse_attributes(text).get(category, ()) if value in allowed)
    return counts.most_common(1)[0][0] if counts else None


def geometry_features(obj):
    polygon = Polygon(obj.contour)
    area = max(float(polygon.area), 1e-6)
    perimeter = max(float(polygon.length), 1e-6)
    rectangle = polygon.minimum_rotated_rectangle
    rectangle_area = max(float(rectangle.area), 1e-6)
    points = np.asarray(rectangle.exterior.coords)[:4]
    edges = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
    aspect = float(edges.max() / max(edges.min(), 1e-6))
    return {
        "area_m2": area, "perimeter_m": perimeter, "aspect_ratio": aspect,
        "rectangularity": area / rectangle_area,
        "compactness": 4 * math.pi * area / (perimeter * perimeter),
    }


def shape_label(features):
    if features["compactness"] >= 0.78:
        return "round"
    if features["rectangularity"] >= 0.78:
        return "square" if features["aspect_ratio"] < 1.35 else "rectangular"
    return "irregular"


def size_thresholds(train_objects):
    areas = np.asarray([geometry_features(obj)["area_m2"] for obj in train_objects])
    return tuple(float(value) for value in np.quantile(areas, (1 / 3, 2 / 3)))


def size_label(area, thresholds):
    return SIZE_CLASSES[0 if area < thresholds[0] else 1 if area < thresholds[1] else 2]


def object_labels(obj, texts, thresholds):
    geometry = geometry_features(obj)
    parsed_context = sorted({value for text in texts for value in parse_attributes(text).get("context", ()) if value in CONTEXT_CLASSES})
    roof_mentions = {value for text in texts for value in parse_attributes(text).get("roof", ())}
    return {
        "color": primary_language_label(texts, "color", COLOR_CLASSES),
        "size": size_label(geometry["area_m2"], thresholds),
        "shape": shape_label(geometry),
        "semantic": obj.object_type if obj.object_type in SEMANTIC_CLASSES else None,
        "context": parsed_context,
        "road_context": "yes" if {"road", "intersection", "crossing"}.intersection(parsed_context) else "no",
        "roof_presence": "yes" if roof_mentions else "no",
        "roof": primary_language_label(texts, "roof", ("flat_roof", "pitched_roof")),
        "geometry": geometry,
    }
