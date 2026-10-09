"""Per-episode immutable CityNav landmark metadata.

Only static landmark contours/names are processed here. Dynamic view masks,
RGB crops and UAV poses must always be refreshed by the environment.
"""
import numpy as np

from multiagent.space import Point2D


def build_static_landmark_observation(nav_map, map_name, map_meters, normalize_position):
    """Match the old per-step normalization, once when an episode resets."""
    contours = nav_map.landmark_map.get_contours()
    centroids = [np.asarray(contour, dtype=np.float64).mean(axis=0)
                 for contour in contours]
    normalized_centroids = [
        normalize_position(Point2D(float(c[0]), float(c[1])), map_name, map_meters)
        for c in centroids
    ]
    refs = []
    landmarks = nav_map.referenced_landmark_map.landmarks
    names = nav_map.referenced_landmark_map.landmark_names
    for name, landmark in zip(names, landmarks):
        if not landmark.contour:
            continue
        norm_contour = np.asarray([
            normalize_position(point, map_name, map_meters)
            for point in landmark.contour
        ], dtype=np.float32)
        if norm_contour.ndim != 2 or norm_contour.shape[1] != 2:
            continue
        if not np.isfinite(norm_contour).all():
            continue
        refs.append({
            'name': name,
            'center_xy': norm_contour.mean(axis=0).tolist(),
            'extent_xy': np.ptp(norm_contour, axis=0).tolist(),
        })
    return {
        'centroids': np.mean(normalized_centroids, axis=0)
                     if normalized_centroids else np.array([0, 0]),
        'centroid_goal': np.mean(centroids, axis=0)
                         if centroids else np.array([0, 0]),
        'reference_landmarks': refs,
    }


def build_uav_landmark_observation(reference_landmarks, position, yaw, map_meters):
    """Known named anchors relative to current UAV, metric east/north axes."""
    result = []
    for landmark in reference_landmarks:
        delta = (np.asarray(landmark['center_xy']) - np.asarray(position)) * map_meters
        east, north = float(delta[0]), float(-delta[1])
        distance = float(np.hypot(east, north))
        result.append(dict(name=landmark['name'], distance_m=distance,
                           east_m=east, north_m=north,
                           relative_bearing_rad=float(np.arctan2(
                               np.sin(np.arctan2(north,east)-yaw),
                               np.cos(np.arctan2(north,east)-yaw)))))
    return result
