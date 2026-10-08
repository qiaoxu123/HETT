"""Structure-only image matching. No XY, scene labels, target or map IDs accepted.
Observed Canny boundaries are unlabeled: not claimed semantic building extraction.
Map outlines are separately rendered by the offline adapter, a map self-match.
"""
import cv2
import numpy as np

SIZE=32
ROTATIONS=(0,90,180,270)
SCALES=(.5,1.,2.)
SHIFTS=(-8,0,8)


def observed_edges(rgb,valid):
    gray=cv2.cvtColor(np.asarray(rgb,dtype=np.uint8),cv2.COLOR_RGB2GRAY)
    gray=cv2.GaussianBlur(gray,(0,0),2)
    edge=cv2.Canny(gray,30,90)
    interior=cv2.erode(valid.astype(np.uint8),np.ones((7,7),np.uint8))
    return (edge>0)&(interior>0)


def small_edges(edge):
    return (cv2.resize(edge.astype(np.float32),(SIZE,SIZE),interpolation=cv2.INTER_AREA)>.08).astype(np.float32)


def distance_field(edge):
    if not np.any(edge):return np.ones(edge.shape,np.float32)
    return np.minimum(cv2.distanceTransform((edge==0).astype(np.uint8),cv2.DIST_L2,3)/8.,1.)


def variants(edge):
    """Fixed image-only rotation/scale/translation bank, no pose alignment."""
    es=[];ds=[]
    for angle in ROTATIONS:
        for scale in SCALES:
            for dx in SHIFTS:
                for dy in SHIFTS:
                    M=cv2.getRotationMatrix2D(((SIZE-1)/2,(SIZE-1)/2),angle,scale)
                    M[:,2]+=[dx,dy]
                    e=cv2.warpAffine(edge,M,(SIZE,SIZE),flags=cv2.INTER_NEAREST)
                    es.append(e/max(e.sum(),1));ds.append(distance_field(e))
    return np.asarray(es).reshape(-1,SIZE*SIZE),np.asarray(ds).reshape(-1,SIZE*SIZE)


def symmetric_chamfer(qedge,cedge):
    if qedge.sum()<3 or cedge.sum()<3:return 0.
    qe=qedge/max(qedge.sum(),1);qd=distance_field(qedge)
    ce,cd=variants(cedge)
    cost=ce@qd.ravel()+cd@qe.ravel()
    cost[ce.sum(-1)==0]=2.
    return float(1-.5*np.min(cost))
