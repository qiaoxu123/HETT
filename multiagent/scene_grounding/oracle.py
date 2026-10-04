"""Oracle scene-evidence scores used to measure the route's upper bound."""
from __future__ import annotations

from .expected_scene import attribute_similarity, count_similarity, landmark_geometry_similarity


def oracle_scores(candidate_scene, target_scene, candidate_xy, target_xy, landmarks):
    attributes = attribute_similarity(candidate_scene, target_scene)
    context = count_similarity(candidate_scene, target_scene, "40")
    geometry = landmark_geometry_similarity(candidate_xy, target_xy, landmarks)
    return {
        "attributes": attributes,
        "context": context,
        "geometry": geometry,
        "scene_geometry": (attributes + context + geometry) / 3.0,
    }
