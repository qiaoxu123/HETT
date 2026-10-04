"""Small, non-learned history baselines."""
from __future__ import annotations

import numpy as np


def fuse_scores(scores, window=1, mode="mean", confidences=None):
    if window < 1:
        raise ValueError("window must be positive")
    values = np.asarray(scores, dtype=float)
    result = []
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1):index + 1]
        if mode == "max":
            result.append(float(chunk.max()))
        elif mode == "confidence":
            weights = np.asarray(confidences[max(0, index - window + 1):index + 1], dtype=float)
            result.append(float(np.average(chunk, weights=np.maximum(weights, 1e-6))))
        elif mode == "mean":
            result.append(float(chunk.mean()))
        else:
            raise ValueError(f"unknown mode: {mode}")
    return result
