"""Split-safe calibration: no val_unseen argument or feature path."""
from __future__ import annotations
import math

def fit_calibration(train_metrics, val_seen_metrics):
    """Choose from shared programs; UNKNOWN takes mass when evidence is weak."""
    shared=set(train_metrics)&set(val_seen_metrics)
    if not shared:return None
    best=max(shared,key=lambda p:(val_seen_metrics[p]['auc'],train_metrics[p]['auc']))
    count=val_seen_metrics[best]['n']
    evidence={p:max(0.,min(train_metrics[p]['auc']-.5,val_seen_metrics[p]['auc']-.5)) for p in shared}
    logits={p:8*evidence[p] for p in shared}
    logits['UNKNOWN']=2. if count<10 else 0.
    z=sum(math.exp(x) for x in logits.values())
    probabilities={p:math.exp(x)/z for p,x in logits.items()}
    reliability=max(0.,min(1.,2*(val_seen_metrics[best]['auc']-.5)))*min(1.,count/20.)
    if train_metrics[best]['auc']<.5:reliability=0.
    return best,probabilities,reliability
