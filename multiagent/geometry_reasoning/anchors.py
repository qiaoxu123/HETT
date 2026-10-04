"""Landmark grounding and auditable geometry records."""
from __future__ import annotations
import re
import numpy as np

def _norm(value): return re.sub(r"[^a-z0-9]", "", value.casefold())

def anchor_record(obj):
    points = np.asarray(obj.contour, float); center = points.mean(0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance); axis = vectors[:, int(np.argmax(values))]
    return {"landmark_id": int(obj.id), "name": obj.name, "object_type": obj.object_type,
            "centroid": [float(obj.position.x), float(obj.position.y)], "contour": points.tolist(),
            "area": float(obj.area), "orientation": [float(axis[0]), float(axis[1])],
            "bbox": [float(points[:,0].min()), float(points[:,1].min()), float(points[:,0].max()), float(points[:,1].max())]}

def resolve_anchors(program, map_objects):
    result = []
    for requested in program.anchors:
        name = requested["name"]; exact = [obj for obj in map_objects if _norm(obj.name) == _norm(name) and obj.name]
        if not exact:
            needle = name.casefold()
            exact = [obj for obj in map_objects if needle in obj.name.casefold() or needle in obj.object_type.casefold()]
        result.append({"query": name, "matches": [anchor_record(obj) for obj in exact]})
    return result
