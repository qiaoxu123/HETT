"""Candidate ordering along an explicit travel axis."""
from __future__ import annotations
import math
import numpy as np

def candidate_order(candidates_xy, origin, axis):
    projections=[float((np.asarray(xy)-np.asarray(origin))@np.asarray(axis)) for xy in candidates_xy]
    order=np.argsort(projections); ranks=np.empty(len(order),int); ranks[order]=np.arange(1,len(order)+1)
    return ranks.tolist(), projections
def ordinal_scores(ranks, requested): return [float(math.exp(-abs(rank-requested))) for rank in ranks]
