"""Image-score rank calibration; alpha selected using val_seen episodes only.
Both inputs are within-query empirical percentile scores across the full fixed
candidate gallery. No GT/coordinates/map ID used in score normalization.
"""
import numpy as np
from scipy.stats import rankdata

ALPHAS=(0.,.25,.5,.75,1.)

def percentile_scores(scores):
    x=np.asarray(scores)
    return (rankdata(x,axis=1,method='average')-.5)/x.shape[1]

def fuse(rgb,geometry,alpha):
    if alpha not in ALPHAS:raise ValueError('alpha outside preregistered grid')
    return alpha*percentile_scores(rgb)+(1-alpha)*percentile_scores(geometry)

def select_alpha(results,split):
    if split!='val_seen':raise ValueError('only val_seen development allowed')
    # Predeclared objective; ties prefer .5 then lower alpha.
    return max(ALPHAS,key=lambda a:(results[str(a)]['hard_accuracy'],-abs(a-.5),-a))
