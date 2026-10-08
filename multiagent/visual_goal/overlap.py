"""Offline orthophoto coverage oracle. Never pass its outputs to a matcher.

Coordinates use each TIFF affine, not MAP_BOUNDS. OpenCV samples pixel centres;
rasterio.index floors to integers. Reproduce both, including that quantization.
Valid support is the convex hull of raster pixel centres: conservative by <= half
one source pixel at the boundary (bilinear interpolation has fractional support).
"""
import cv2
import numpy as np
from shapely.geometry import Polygon


def pose_corners(xy, height, yaw):
    if height <= 0:
        raise ValueError('nonpositive height')
    front = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-np.sin(yaw), np.cos(yaw)])
    return np.asarray(xy) + height * np.array([front+left, front-left, -front-left, -front+left])


def crop_transform(transform, width, height, xy, agl, yaw, size=224):
    corners = pose_corners(xy, agl, yaw)
    inv = ~transform
    src = np.floor([inv * tuple(p) for p in corners]).astype(np.float32)
    dst = np.float32([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]])
    H = cv2.getPerspectiveTransform(src, dst)
    effective = np.asarray([transform * (float(x)+.5, float(y)+.5) for x, y in src])
    boundary = Polygon([transform * p for p in ((.5,.5),(width-.5,.5),(width-.5,height-.5),(.5,height-.5))])
    full = Polygon(effective)
    valid = full.intersection(boundary)
    return H, full, valid


def world_to_image(points, transform, H):
    inv = ~transform
    p = np.asarray(points, dtype=float)
    source = np.float32(np.stack([inv.a*p[:,0]+inv.b*p[:,1]+inv.c-.5, inv.d*p[:,0]+inv.e*p[:,1]+inv.f-.5], axis=-1))
    return cv2.perspectiveTransform(source.reshape(-1, 1, 2), H).reshape(-1, 2)


def valid_mask(H, width, height, size=224):
    yy, xx = np.mgrid[:size, :size]
    src = cv2.perspectiveTransform(np.float32(np.stack([xx, yy], -1)).reshape(-1,1,2), np.linalg.inv(H)).reshape(size,size,2)
    return (src[...,0]>=0)&(src[...,0]<=width-1)&(src[...,1]>=0)&(src[...,1]<=height-1)


def coverage(query, template):
    area = query.intersection(template).area
    # Negligible polygon arithmetic slivers, below 1e-6 m², count as zero.
    if area < 1e-6:
        area = 0.
    qa, ta = query.area, template.area
    return dict(intersection_m2=area, query_coverage=area/qa if qa else 0.,
                template_coverage=area/ta if ta else 0., iou=area/(qa+ta-area) if qa+ta>area else 0.)


BINS = [(0.,0.,'0%'), (0.,.1,'(0,10]%'), (.1,.3,'(10,30]%'), (.3,.5,'(30,50]%'), (.5,.7,'(50,70]%'), (.7,1.000001,'(70,100]%')]

def bin_mask(values, lo, hi):
    x = np.asarray(values)
    return x == 0 if hi == 0 else ((x > lo) & (x <= hi))
