"""Primitive constraint scores in a resolved reference frame."""
from __future__ import annotations
import math
import numpy as np

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def relation_score(relation,points,anchor,axis,sigma=20.):
    p=np.asarray(points,float);a=np.asarray(anchor,float);h=np.asarray(axis,float);right=np.asarray([h[1],-h[0]]);off=p-a
    if relation=="right_of":return sigmoid((off@right)/sigma)
    if relation=="left_of":return sigmoid(-(off@right)/sigma)
    if relation in ("front_of","after","past"):return sigmoid((off@h)/sigma)
    if relation in ("behind","before"):return sigmoid(-(off@h)/sigma)
    if relation=="near":return np.exp(-np.linalg.norm(off,axis=1)/30.)
    if relation=="along":return np.exp(-np.abs(off@right)/20.)
    return np.full(len(p),.5)

def between_score(points,a,b,sigma=20.):
    p=np.asarray(points,float);a=np.asarray(a,float);b=np.asarray(b,float);v=b-a;den=float(v@v)
    if den<1e-8:return np.zeros(len(p))
    t=((p-a)@v)/den;proj=a+np.clip(t,0,1)[:,None]*v;return np.exp(-np.linalg.norm(p-proj,axis=1)/sigma)*sigmoid(10*t)*sigmoid(10*(1-t))*4
