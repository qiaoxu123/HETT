"""GT-free goal stabilization for bounded waypoint execution.

Re-planning every step lets the executed goal jitter or jump to another
candidate after the UAV already reached the right place (OSR >> SR). These
rules only use the model's own scores and the UAV pose:

* arrival lock: once the UAV is within ``lock_m`` of its current goal, keep it;
* switch margin: switch to a new goal cell only if its score beats the score
  of the previous goal cell (under the CURRENT belief) by ``margin`` nats.

Both are disabled by default (lock_m <= 0, margin <= 0), which reproduces the
original behaviour exactly.
"""
from __future__ import annotations

import numpy as np


def stabilize_goals(prev_xy, prev_cell, new_xy, new_cell, score_maps,
                    dist_to_prev_m, *, lock_m=0.0, margin=0.0):
    """Return (goal_xy [B,2], goal_cell [B]) after lock / hysteresis.

    prev_cell < 0 marks episodes without a previous goal. score_maps is
    [B, cells] log-scores (``-inf`` where a cell is not a candidate).
    """
    out_xy = np.array(new_xy, dtype=np.float32, copy=True)
    out_cell = np.array(new_cell, dtype=np.int64, copy=True)
    if lock_m <= 0 and margin <= 0:
        return out_xy, out_cell
    for i in range(len(out_cell)):
        if prev_cell[i] < 0:
            continue
        # Locked goals are frozen exactly (also against sub-cell jitter of
        # refined coordinates); otherwise the same cell needs no decision.
        keep = lock_m > 0 and dist_to_prev_m[i] <= lock_m
        if not keep and prev_cell[i] == new_cell[i]:
            continue
        if not keep and margin > 0:
            prev_score = score_maps[i, prev_cell[i]]
            new_score = score_maps[i, new_cell[i]]
            keep = np.isfinite(prev_score) and new_score - prev_score < margin
        if keep:
            out_xy[i] = prev_xy[i]
            out_cell[i] = prev_cell[i]
    return out_xy, out_cell
