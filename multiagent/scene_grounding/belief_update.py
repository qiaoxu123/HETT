"""Transparent additive evidence updates over a fixed B0 candidate pool."""
from __future__ import annotations

import numpy as np


def additive_candidate_update(b0, visual, geometry=None, lambda_visual=1.0, lambda_geometry=1.0, clip=2.0):
    prior = np.asarray(b0, dtype=float)
    visual = np.clip(np.asarray(visual, dtype=float), -clip, clip)
    geometry = np.zeros_like(visual) if geometry is None else np.clip(np.asarray(geometry, dtype=float), -clip, clip)
    logits = np.log(np.maximum(prior, 1e-12)) + lambda_visual * visual + lambda_geometry * geometry
    logits -= logits.max()
    posterior = np.exp(logits); posterior /= posterior.sum()
    return posterior


def temporal_additive_update(b0, evidence_frames, *, decay=0.8, clip=2.0):
    evidence = np.zeros_like(np.asarray(b0, dtype=float))
    for frame in evidence_frames:
        evidence = decay * evidence + np.clip(np.asarray(frame, dtype=float), -clip, clip)
    return additive_candidate_update(b0, evidence, lambda_visual=1.0, clip=clip)
