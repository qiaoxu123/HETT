"""Small evidence-limited spatial constraint graph for later use."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

ALLOWED=frozenset({'NEAR','NEXT_TO','ROAD_ASSOCIATION','FOOTPRINT_DISTANCE',
                   'LAYOUT_ORDER','ROW_INDEX','COLUMN_INDEX','ROAD_SEQUENCE'})
GENERIC_DIRECTION=frozenset({'BEHIND','FRONT','LEFT_OF','RIGHT_OF'})

@dataclass(frozen=True)
class Constraint:
    kind:str
    anchor_phrase:str|None=None
    explicit_frame:bool=False

def admissible(edge:Constraint)->bool:
    return edge.kind in ALLOWED or (edge.kind in GENERIC_DIRECTION and edge.explicit_frame)

def combine(layout,anchor,lambda_layout=.5):
    """Two separate score channels; no answer-dependent feature is accepted."""
    if not 0<=lambda_layout<=1:raise ValueError('lambda_layout must be in [0,1]')
    x=np.asarray(layout,float);y=np.asarray(anchor,float)
    if x.shape!=y.shape:raise ValueError('score shapes differ')
    return lambda_layout*x+(1-lambda_layout)*y

def calibrate_lambda(train_seen,val_seen):
    """Grid search over validation rows; API deliberately has no unseen input."""
    if not train_seen or not val_seen:raise ValueError('train_seen and val_seen required')
    options=np.linspace(0,1,11)
    def objective(rows,lam):
        correct=0.
        for layout,anchor,target_index in rows:
            s=combine(layout,anchor,lam)
            correct+=(1/np.sum(np.isclose(s,np.max(s))) if np.isclose(s[int(target_index)],np.max(s)) else 0.)
        return correct/len(rows)
    train_best=max(options,key=lambda x:objective(train_seen,x))
    # Validation selects among nearby train-supported options, never unseen.
    supported=[x for x in options if abs(x-train_best)<=.2+1e-9]
    return float(max(supported,key=lambda x:objective(val_seen,x)))
