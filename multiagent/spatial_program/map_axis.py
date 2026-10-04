"""Principal-axis extraction from mapped contours."""
from __future__ import annotations
import numpy as np

def principal_axis(points):
    pts=np.asarray(points,float)
    if len(pts)<2:return np.asarray([1.,0.]),0.
    centered=pts-pts.mean(0);values,vectors=np.linalg.eigh(centered.T@centered/max(len(pts),1));order=np.argsort(values)
    axis=vectors[:,order[-1]];confidence=float((values[order[-1]]-values[order[-2]])/max(values[order[-1]],1e-9))
    if axis[0]<0:axis=-axis
    return axis,float(np.clip(confidence,0,1))

def local_road_axis(objects,center,radius=60.):
    center=np.asarray(center,float);points=[]
    for obj in objects:
        if "road" in obj.object_type.casefold() or "street" in obj.object_type.casefold():
            pts=np.asarray(obj.contour,float)
            points.extend(pts[np.linalg.norm(pts-center,axis=1)<=radius].tolist())
    return principal_axis(points) if len(points)>=2 else (np.asarray([1.,0.]),0.)
