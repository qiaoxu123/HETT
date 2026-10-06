from __future__ import annotations

import math
import cv2
import numpy as np


def project_world_polygon(contour_xy, pose5d, *, image_size=(224, 224), fov_deg=90.0,
                          object_z=0.0, object_height=0.0):
    """Project CityNav ENU polygon vertices onto a forward camera image.

    ``pose5d`` is (x,y,z,yaw,pitch), with positive pitch denoting upward in
    the dataset convention; ``view_pitch_deg`` is supplied through the pose
    pitch by the caller. Returns a raster mask and projected/visible geometry
    metadata. This is a pinhole geometry projection, not semantic segmentation.
    """
    h, w = map(int, image_size)
    yaw, pitch = float(pose5d[3]), float(pose5d[4])
    px, py, pz = map(float, pose5d[:3])
    points = np.asarray(contour_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return None
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([math.sin(yaw), -math.cos(yaw)])
    delta = points - np.array([px, py])
    along = delta @ forward
    lateral = delta @ right
    z = float(object_z) + max(0.0, float(object_height))
    dz = z - pz
    # Camera pitch follows the source Pose5D convention: negative is down.
    down_angle = -pitch
    cp, sp = math.cos(down_angle), math.sin(down_angle)
    depth = cp * along - sp * dz
    vertical_down = sp * along + cp * (-dz)
    valid = depth > 0.25
    # A polygon crossing the near plane cannot be reliably rasterized from its
    # vertices alone; mark it unobservable instead of drawing an artificial ROI.
    if not np.all(valid):
        return None
    f = (w / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
    uv = np.stack([w / 2.0 + f * lateral / depth,
                   h / 2.0 + f * vertical_down / depth], axis=1)
    raw_area = abs(float(cv2.contourArea(uv.astype(np.float32))))
    # Bound work for very close, partly off-screen polygons.
    uv = np.nan_to_num(uv, nan=-float(w), posinf=10.0 * w, neginf=-10.0 * w)
    clipped = np.rint(np.clip(uv, -10.0 * max(w, h), 10.0 * max(w, h))).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [clipped], 1)
    pixel_area = int(mask.sum())
    if pixel_area == 0:
        return None
    yy, xx = np.where(mask)
    return {
        "mask": mask,
        "bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
        "pixel_area": pixel_area,
        "visible_ratio": float(min(pixel_area / max(raw_area, 1.0), 1.0)),
        "center_uv": [float(xx.mean()), float(yy.mean())],
        "in_front_fraction": float(valid.mean()),
        "projection": "pinhole_citynav_world_geometry_no_occlusion",
    }
