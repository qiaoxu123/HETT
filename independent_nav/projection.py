"""Top-down camera geometry independent of HETT."""

import math

import cv2
import numpy as np
from shapely.geometry import Polygon


def view_corners(x, y, yaw, altitude):
    """Return front-left, front-right, back-right, back-left world corners."""
    front = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    left = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    center = np.asarray([x, y], dtype=np.float64)
    return np.asarray([
        center + altitude * (front + left),
        center + altitude * (front - left),
        center + altitude * (-front - left),
        center + altitude * (-front + left),
    ], dtype=np.float32)


def world_to_image_homography(corners, image_size):
    destination = np.asarray([
        [0, 0], [image_size - 1, 0],
        [image_size - 1, image_size - 1], [0, image_size - 1],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(
        np.asarray(corners, dtype=np.float32), destination)


def transform_points(points, homography):
    values = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(values, homography)[0]


def image_to_world(points, world_to_image):
    return transform_points(points, np.linalg.inv(world_to_image))


def valid_polygon(points):
    polygon = Polygon(points)
    return polygon if polygon.is_valid else polygon.buffer(0)


def rasterize_polygon(geometry, homography, image_size):
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    if geometry.is_empty:
        return mask
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for polygon in polygons:
        if polygon.geom_type != "Polygon" or polygon.is_empty:
            continue
        points = np.asarray(polygon.exterior.coords[:-1], dtype=np.float32)
        pixels = np.round(transform_points(points, homography)).astype(np.int32)
        if len(pixels) >= 3:
            cv2.fillPoly(mask, [pixels], 1)
    return mask
