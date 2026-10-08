"""Leakage-separated template retrieval and similarity diagnostics."""
from __future__ import annotations

import numpy as np

from .metrics import retrieval_metrics


def evaluate_template_retrieval(query_features, candidate_features, query_meta, candidate_meta):
    # Encoder interfaces receive arrays of images only. Instance IDs, positions,
    # candidate order and episode labels are introduced strictly after encoding.
    return retrieval_metrics(
        query_features, candidate_features,
        [r["scene_key"] for r in query_meta], [r["scene_key"] for r in candidate_meta],
        [r["map_name"] for r in query_meta], [r["map_name"] for r in candidate_meta],
        query_distances=[r["distance_to_goal_m"] for r in query_meta],
        query_episode_ids=[r["episode_key"] for r in query_meta],
    )


def choose_distance_stratum(poses, target_xy, bounds=((80, float("inf")), (40, 80), (20, 40), (0, 20))):
    """At most one real trajectory pose per requested range; never synthesize."""
    target = np.asarray(target_xy, dtype=float)
    xy = np.asarray([[p[0], p[1]] for p in poses], dtype=float)
    dist = np.linalg.norm(xy - target[None, :], axis=1)
    result = []
    for lo, hi in bounds:
        inds = np.where((dist >= lo) & (dist < hi))[0]
        if len(inds):
            # Pick a deterministic real pose nearest the band's midpoint.
            midpoint = lo + 10 if not np.isfinite(hi) else (lo + hi) / 2
            chosen = int(inds[np.argmin(np.abs(dist[inds] - midpoint))])
            result.append((chosen, float(dist[chosen])))
    return result


def paired_hard_negative_indices(query, candidate, query_meta, candidate_meta, mode="same_map"):
    """Select negatives without exposing identity labels to the encoder."""
    q = _unit(np.asarray(query))
    c = _unit(np.asarray(candidate))
    scores = q @ c.T
    out = []
    for i, meta in enumerate(query_meta):
        allowed = np.asarray([
            x["scene_key"] != meta["scene_key"] and
            (mode != "same_map" or x["map_name"] == meta["map_name"])
            for x in candidate_meta
        ])
        if allowed.any():
            row = np.where(allowed, scores[i], -np.inf)
            out.append(int(row.argmax()))
        else:
            out.append(-1)
    return out


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

