"""Continuous analytic relation scores."""
from __future__ import annotations
import math
import numpy as np

def sigmoid(value): return float(1 / (1 + math.exp(-float(np.clip(value, -30, 30)))))
def unit(vector, fallback=(1.0, 0.0)):
    value=np.asarray(vector,float); norm=np.linalg.norm(value); return value/norm if norm>1e-6 else np.asarray(fallback,float)

def reference_frame(anchor_xy, pose_xy, yaw, anchor_orientation=None):
    delta=np.asarray(anchor_xy)-np.asarray(pose_xy)
    if np.linalg.norm(delta)>1e-6: return unit(delta), "pose_to_anchor"
    if yaw is not None: return np.asarray([math.cos(yaw), math.sin(yaw)]), "pose_yaw"
    if anchor_orientation is not None: return unit(anchor_orientation), "anchor_orientation"
    return np.asarray([1.,0.]), "map_north_up"

def directional(candidate, anchor, heading, kind, sigma=20.0):
    offset=np.asarray(candidate)-np.asarray(anchor); right=np.asarray([heading[1],-heading[0]])
    projection = float(offset@right) if kind in ("left_of","right_of") else float(offset@heading)
    if kind in ("left_of","behind"): projection=-projection
    return sigmoid(projection/max(sigma,1e-6))
def near(candidate, anchor, sigma=30.0): return float(math.exp(-np.linalg.norm(np.asarray(candidate)-np.asarray(anchor))/sigma))
def far(candidate, anchor, sigma=60.0): return float(1-math.exp(-np.linalg.norm(np.asarray(candidate)-np.asarray(anchor))/sigma))
def sequence(candidate, anchor, heading, kind, sigma=20.0):
    along=float((np.asarray(candidate)-np.asarray(anchor))@heading)
    return sigmoid((along if kind in ("after","past") else -along)/sigma)
def between(candidate, left, right, sigma=20.0):
    a,b,p=map(lambda x:np.asarray(x,float),(left,right,candidate)); segment=b-a; denom=float(segment@segment)
    if denom<1e-6:return 0.0
    t=float((p-a)@segment/denom); projection=a+np.clip(t,0,1)*segment; perpendicular=np.linalg.norm(p-projection)
    range_score=sigmoid(10*t)*sigmoid(10*(1-t)); return float(range_score*math.exp(-perpendicular/sigma)*4)
