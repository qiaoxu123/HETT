from __future__ import annotations

import numpy as np


def rerank(belief_scores: np.ndarray, visual_scores: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Return candidate indices ordered by visual rerank score within a belief shortlist."""
    b, v = np.asarray(belief_scores), np.asarray(visual_scores)
    if b.shape != v.shape:
        raise ValueError("belief and visual score arrays must have the same shape")
    k = min(len(b), max(1, int(np.sqrt(len(b)))))
    shortlist = np.argsort(-b)[:k]
    return shortlist[np.argsort(-(b[shortlist] + alpha * v[shortlist]))]
