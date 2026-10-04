"""Visual evidence containers and interpretable template matching."""
from __future__ import annotations

import math
from dataclasses import dataclass


SEMANTIC_MAP = {
    "building": "Building", "house": "Building", "church": "Building",
    "library": "Building", "college": "Building", "school": "Building",
    "office": "Building", "tower": "Building", "hospital": "Building",
    "car": "Car", "parking": "Parking", "parking_lot": "Parking",
    "field": "Ground", "ground": "Ground", "grass": "Ground",
}


@dataclass(frozen=True)
class SceneEvidence:
    whole: dict
    regions: dict
    combined: dict
    region_count: int = 0

    def representation(self, name: str) -> dict:
        if name not in {"whole", "regions", "combined"}:
            raise ValueError(f"unknown representation: {name}")
        return getattr(self, name)


def _probability(evidence: dict, category: str, value: str) -> float | None:
    if category == "context":
        if value in {"road", "intersection", "crossing"}:
            category, value = "road_context", "yes"
        elif value == "parking_lot":
            category, value = "semantic", "Parking"
        elif value in {"field", "grass", "vegetation", "open_area"}:
            category, value = "semantic", "Ground"
        else:
            return None
    elif category == "semantic":
        value = SEMANTIC_MAP.get(value, value)
    elif category == "roof":
        category, value = "roof_presence", "yes"
    values = evidence.get(category, {})
    return float(values[value]) if value in values else None


def explicit_match(template, evidence: dict, component: str = "all") -> dict:
    """Return an average log probability and auditable per-clause evidence."""
    clauses = []
    requirements = template.visual_requirements
    for category, values in requirements.items():
        group = "anchor" if category == "context" else "target"
        if component not in {"all", group, category}:
            continue
        for value in values:
            probability = _probability(evidence, category, value)
            if probability is not None:
                clauses.append({"group": group, "category": category, "value": value,
                                "probability": probability})
    if not clauses:
        return {"score": 0.0, "coverage": 0, "clauses": []}
    score = sum(math.log(max(item["probability"], 1e-6)) for item in clauses) / len(clauses)
    return {"score": float(score), "coverage": len(clauses), "clauses": clauses}


def combine_probabilities(whole: dict, region_predictions: list[dict]) -> SceneEvidence:
    regions = {}
    for category in set().union(*(prediction.keys() for prediction in region_predictions)) if region_predictions else ():
        labels = set().union(*(prediction.get(category, {}).keys() for prediction in region_predictions))
        regions[category] = {label: max(prediction.get(category, {}).get(label, 0.0)
                                         for prediction in region_predictions) for label in labels}
    combined = {}
    for category in set(whole) | set(regions):
        # Global context is a scene property; target appearance benefits from localized regions.
        if category == "road_context":
            combined[category] = whole.get(category, regions.get(category, {}))
        elif category in regions:
            combined[category] = regions[category]
        else:
            combined[category] = whole[category]
    return SceneEvidence(whole=whole, regions=regions, combined=combined,
                         region_count=len(region_predictions))
