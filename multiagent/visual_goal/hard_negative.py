from __future__ import annotations

import numpy as np


def select_negatives(query: dict, templates: list[dict]) -> dict[str, int | None]:
    """Indices for random, same-map, nearby, visually-similar negatives."""
    candidates = [(i, t) for i, t in enumerate(templates) if t.get("episode_id") != query.get("episode_id")]
    def nearest(rows, key):
        return min(rows, key=key)[0] if rows else None
    same = [(i,t) for i,t in candidates if t.get("map_name") == query.get("map_name")]
    xy = np.asarray(query.get("target_xy", (0,0)), dtype=float)
    return {
        "random": candidates[0][0] if candidates else None,
        "same_map": same[0][0] if same else None,
        "nearby": nearest(same, lambda it: np.linalg.norm(np.asarray(it[1].get("target_xy", (0,0))) - xy)),
        "visually_similar": None,  # filled by feature-space nearest-neighbor selection
    }
