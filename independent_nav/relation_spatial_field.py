"""Target-free spatial fields derived from a named landmark boundary.

The functions in this module deliberately have no target-coordinate argument.
They turn one parsed relation and one map landmark boundary into scores for
arbitrary query points (grid cells or generated candidate endpoints).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np


MAP_METERS = 410.0

# Selected on train_seen by the existing reference-frame audit.  It maps a
# language-frame direction into the north-up normalized map frame.
LANGUAGE_TO_MAP = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

DIRECTION_VECTORS = {
    "left": np.asarray([-1.0, 0.0], dtype=np.float32),
    "right": np.asarray([1.0, 0.0], dtype=np.float32),
    "above": np.asarray([0.0, -1.0], dtype=np.float32),
    "below": np.asarray([0.0, 1.0], dtype=np.float32),
}

DIRECTION_PATTERNS = {
    "left": re.compile(r"(?:to|on) the left (?:side )?of (?:the )?masktoken"),
    "right": re.compile(r"(?:to|on) the right (?:side )?of (?:the )?masktoken"),
    "above": re.compile(r"(?:above|north of|north side of) (?:the )?masktoken"),
    "below": re.compile(r"(?:below|south of|south side of|under) (?:the )?masktoken"),
}

# A near cue is accepted only when it directly governs the masked landmark.
# This excludes descriptions such as "cars parked near it" that happen to
# contain an unrelated landmark elsewhere in the sentence.
NEAR_PATTERNS = (
    re.compile(
        r"(?:near|next to|beside|adjacent to|close to|alongside|by) "
        r"(?:the )?masktoken"
    ),
    re.compile(r"masktoken (?:building |road |street )?(?:nearby|close by)"),
)

FLIPPED_DIRECTION = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}


def _exact_mask(description: str, value: str) -> tuple[str, bool]:
    words = re.findall(r"[A-Za-z0-9]+", value or "")
    if not words:
        return description, False
    pattern = r"\b" + r"\W+".join(re.escape(word) for word in words) + r"\b"
    masked, count = re.subn(
        pattern, "[MASK]", description, count=1, flags=re.IGNORECASE
    )
    return masked, bool(count)


def mask_landmark_span(
    description: str,
    query: str,
    resolved_name: str | None = None,
    fuzzy_threshold: float = 0.58,
) -> tuple[str, bool]:
    """Mask an exact/alias/fuzzy landmark span without retaining its name."""
    for value in (query, resolved_name):
        if value:
            masked, found = _exact_mask(description, value)
            if found:
                return masked, True

    query_words = re.findall(r"[a-z0-9]+", (query or "").lower())
    matches = list(re.finditer(r"[A-Za-z0-9]+", description))
    if not query_words or not matches:
        return description, False
    query_key = "".join(query_words)
    best = (0.0, None, None)
    for length in range(max(1, len(query_words) - 2), len(query_words) + 3):
        for start in range(0, len(matches) - length + 1):
            end = start + length
            candidate = "".join(
                match.group(0).lower() for match in matches[start:end]
            )
            ratio = SequenceMatcher(None, query_key, candidate).ratio()
            if ratio > best[0]:
                best = ratio, matches[start].start(), matches[end - 1].end()
    if best[0] < fuzzy_threshold:
        return description, False
    return description[: best[1]] + "[MASK]" + description[best[2] :], True


def extract_relation(masked_description: str) -> dict[str, object]:
    """Return strict directional cues and landmark-anchored near presence."""
    normalized = re.sub(
        r"\s+", " ", masked_description.lower().replace("[mask]", "masktoken")
    )
    directions = [
        name for name, pattern in DIRECTION_PATTERNS.items() if pattern.search(normalized)
    ]
    return {
        "directions": directions,
        "near": any(pattern.search(normalized) for pattern in NEAR_PATTERNS),
    }


def nearest_boundary_delta(
    points: np.ndarray, boundary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return point-minus-nearest-boundary deltas and distances."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    boundary = np.asarray(boundary, dtype=np.float32).reshape(-1, 2)
    if not len(boundary):
        raise ValueError("landmark boundary must contain at least one point")
    deltas = points[:, None, :] - boundary[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    nearest = distances.argmin(axis=-1)
    chosen = deltas[np.arange(len(points)), nearest]
    return chosen, distances[np.arange(len(points)), nearest]


def directional_field(
    points: np.ndarray, boundary: np.ndarray, relation: str
) -> np.ndarray:
    """Score whether query points lie in the stated landmark-relative direction."""
    if relation not in DIRECTION_VECTORS:
        raise ValueError(f"unsupported directional relation: {relation}")
    deltas, distances = nearest_boundary_delta(points, boundary)
    unit = deltas / np.maximum(distances[:, None], 1e-6)
    expected_map = LANGUAGE_TO_MAP @ DIRECTION_VECTORS[relation]
    # [-1, 1] cosine agreement is mapped to a calibrated-free [0, 1] field.
    return np.clip((unit @ expected_map + 1.0) * 0.5, 0.0, 1.0)


def near_field(
    points: np.ndarray, boundary: np.ndarray, scale_m: float = 30.0
) -> np.ndarray:
    """Boundary-aware proximity field; large landmarks are not reduced to a center."""
    if scale_m <= 0:
        raise ValueError("scale_m must be positive")
    _, distances = nearest_boundary_delta(points, boundary)
    return np.exp(-(distances * MAP_METERS) / scale_m)


def grid_points(grid_size: int = 20) -> np.ndarray:
    axis = (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)
