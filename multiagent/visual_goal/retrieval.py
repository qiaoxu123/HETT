from __future__ import annotations

import numpy as np


def cosine_matrix(query_features: np.ndarray, template_features: np.ndarray) -> np.ndarray:
    q = np.asarray(query_features, dtype=np.float32)
    t = np.asarray(template_features, dtype=np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return q @ t.T


def same_map_mask(query_maps: list[str], template_maps: list[str]) -> np.ndarray:
    return np.asarray([[qm == tm for tm in template_maps] for qm in query_maps], dtype=bool)
