"""Scoped ordinal operations over already filtered candidates."""
from __future__ import annotations
import numpy as np

def ordinal_scores(points,base_scores,axis,origin,index,temperature=.7):
    points=np.asarray(points,float);base=np.asarray(base_scores,float);axis=np.asarray(axis,float);origin=np.asarray(origin,float)
    order=np.argsort((points-origin)@axis)
    # Filter scope is soft: order candidates by projection, breaking ties using pre-ordinal score.
    projection=(points-origin)@axis
    order=np.lexsort((-base,projection));rank=np.empty(len(points),int);rank[order]=np.arange(1,len(points)+1)
    score=np.exp(-np.abs(rank-index)/max(temperature,1e-6))
    return score,{"order":order.tolist(),"rank":rank.tolist(),"selected":int(order[min(max(index-1,0),len(order)-1)]) if len(order) else None}
