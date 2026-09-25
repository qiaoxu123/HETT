"""Map-conditioned landmark projection for the independent navigator."""

import cv2
import numpy as np

from independent_nav.pointcloud import camera_basis


def project_landmark_prism(contour, position, dimension, pose, image_shape,
                           hfov_deg=90.0):
    """Project a map building's annotated 3D prism into the current camera."""
    height, width = image_shape
    contour = np.asarray(contour, dtype=np.float64)
    center_z = float(position[2])
    half_height = max(float(dimension[2]) / 2.0, 0.5)
    bottom_z, top_z = center_z - half_height, center_z + half_height
    bottom = np.column_stack((contour, np.full(len(contour), bottom_z)))
    top = np.column_stack((contour, np.full(len(contour), top_z)))
    vertices = np.concatenate((bottom, top))

    right, up, forward = camera_basis(pose[3:6])
    relative = vertices - np.asarray(pose[:3], dtype=np.float64)
    depth = relative @ forward
    focal = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    projected = np.column_stack((
        width / 2.0 + focal * (relative @ right) / np.maximum(depth, 1e-6),
        height / 2.0 - focal * (relative @ up) / np.maximum(depth, 1e-6),
    ))
    mask = np.zeros((height, width), dtype=np.uint8)
    count = len(contour)
    faces = [np.arange(count), np.arange(count, 2 * count)]
    for index in range(count):
        following = (index + 1) % count
        faces.append(np.asarray([index, following, following + count, index + count]))
    for face in faces:
        if np.all(depth[face] > 0.5):
            cv2.fillPoly(mask, [np.rint(projected[face]).astype(np.int32)], 1)
    return mask.astype(bool)
