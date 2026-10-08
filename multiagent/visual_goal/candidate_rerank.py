"""Offline candidate reranking primitives; does not call a controller."""
from __future__ import annotations

import numpy as np


def rerank_topk(belief_scores, visual_scores, candidate_xy, target_xy, alpha):
    """Fuse a frozen belief prior and visual similarity; labels used only for metrics."""
    belief = np.asarray(belief_scores, dtype=np.float64)
    visual = np.asarray(visual_scores, dtype=np.float64)
    xy = np.asarray(candidate_xy, dtype=np.float64)
    if belief.shape != visual.shape or len(belief) != len(xy):
        raise ValueError("belief, visual scores and candidate coordinates must align")
    if not 0 <= float(alpha) <= 1:
        raise ValueError("alpha must be in [0,1]")
    fused = (1.0 - alpha) * _zscore(belief) + alpha * _zscore(visual)
    order = np.argsort(-fused, kind="stable")
    return {
        "order": order,
        "top1_index": int(order[0]),
        "top1_distance_m": float(np.linalg.norm(xy[order[0]] - np.asarray(target_xy))),
        "rerank_scores": fused,
    }


def candidate_recall(candidate_xy, target_xy, ks=(1, 4, 8, 16), tolerance_m=20.0):
    xy = np.asarray(candidate_xy, dtype=np.float64)
    d = np.linalg.norm(xy - np.asarray(target_xy, dtype=np.float64), axis=-1)
    return {f"R@{k}/{len(d)}": bool(np.any(d[:min(k, len(d))] <= tolerance_m)) for k in ks}


def oracle_order(candidate_xy, target_xy):
    """Impossible-best geometry ordering, for an upper-bound diagnostic only."""
    xy = np.asarray(candidate_xy, dtype=np.float64)
    return np.argsort(np.linalg.norm(xy - np.asarray(target_xy, dtype=np.float64), axis=-1), kind="stable")


def _zscore(values):
    x = np.asarray(values, dtype=np.float64)
    return (x - x.mean()) / max(float(x.std()), 1e-8)

