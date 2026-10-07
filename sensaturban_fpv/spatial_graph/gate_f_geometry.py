"""Gate F geometry. All feature names and validity rules live in this file.

The functions accept geometry only; none receives a target label, target ID,
candidate rank or answer position. Callers may use labels *after* scoring to
compute diagnostic metrics.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

EPS = 1e-8
WINDOWS = (3, 5, 10, 20)
FRAME_NAMES = ('GLOBAL', 'START', 'FINAL3', 'FINAL5', 'FINAL10', 'FINAL20',
               'ROUTE_END', 'ROUTE_PCA', 'ROAD', 'ANCHOR_MAJOR', 'ANCHOR_MINOR')
GEOMETRY_COLUMNS = (
    'center_distance', 'footprint_distance', 'dx', 'dy',
    'road_distance', 'road_corridor', 'road_tangent_coord', 'road_normal_coord',
    'road_side_sign', 'road_projection', 'road_end_distance',
    'road_crosses', 'road_opposite_side',
    'between_t', 'between_perpendicular', 'between_distance_a',
    'between_distance_b', 'between_balance', 'between_anchor_separation',
    'between_inside_segment',
)
GEOMETRY_INDEX = {name: i for i, name in enumerate(GEOMETRY_COLUMNS)}


def unit(x):
    a = np.asarray(x, dtype=float)[:2]
    n = float(np.linalg.norm(a))
    return a / n if np.isfinite(n) and n > EPS else None


def canonical_axis(x):
    a = unit(x)
    if a is None: return None
    return a if a[0] > 0 or (abs(a[0]) < EPS and a[1] >= 0) else -a


def perpendicular(axis):
    a = unit(axis)
    return None if a is None else np.array([-a[1], a[0]])


def trajectory_frames(trajectory, min_motion=0.5):
    """Window starts at step T-N; motion rather than the last noisy look yaw."""
    tr = np.asarray(trajectory, dtype=float)
    out = {'trajectory_xyz': tr[:, :3].tolist(),
           'trajectory_heading': [unit(v).tolist() if unit(v) is not None else None for v in tr[:, 3:5]],
           'start_position': tr[0, :3].tolist() if len(tr) else None,
           'start_heading': unit(tr[0, 3:5]).tolist() if len(tr) and unit(tr[0, 3:5]) is not None else None,
           'heading_stability': {}}
    for n in WINDOWS:
        key = f'final_heading_{n}'
        if len(tr) < 2:
            out[key] = None; out['heading_stability'][str(n)] = None; continue
        start = max(0, len(tr) - n)
        motion = tr[-1, :2] - tr[start, :2]
        heading = unit(motion) if np.linalg.norm(motion) >= min_motion else None
        out[key] = heading.tolist() if heading is not None else None
        steps = np.diff(tr[start:, :2], axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        moving = steps[lengths >= min_motion]
        out['heading_stability'][str(n)] = (float(np.mean((moving / np.linalg.norm(moving, axis=1)[:, None]) @ heading))
                                                  if heading is not None and len(moving) else None)
    if len(tr) >= 2:
        end = unit(tr[-1, :2] - tr[0, :2]) if np.linalg.norm(tr[-1, :2] - tr[0, :2]) >= min_motion else None
        xy = tr[:, :2] - tr[:, :2].mean(axis=0)
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        pca = unit(vt[0]) if len(vt) else None
        if pca is not None and end is not None and pca @ end < 0: pca = -pca
        elif pca is not None and end is None: pca = canonical_axis(pca)
    else: end = pca = None
    out['route_heading'] = end.tolist() if end is not None else None
    out['route_pca_heading'] = pca.tolist() if pca is not None else None
    return out


def anchor_axes(footprint):
    pts = np.asarray(footprint, dtype=float)
    if pts.ndim != 2 or len(pts) < 2: return None, None
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    if np.linalg.norm(xy) < EPS: return None, None
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    major = canonical_axis(vt[0])
    return major, perpendicular(major)


@dataclass
class RoadProjection:
    distance: float
    nearest_point: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    side: float
    projection: float
    end_distance: float
    corridor: bool


def road_projection(point, region, members, buffer_m=8.0):
    """Use the closest local segment of a RoadRegion polyline, never its global axis."""
    from .road_region import region_polyline
    from .edge_features import polygon_of
    from shapely.geometry import Point
    p = np.asarray(point, dtype=float)[:2]
    polyline, fallback = region_polyline(members) if members else (np.asarray(region.center[:2])[None, :], region.major_axis)
    if len(polyline) < 2:
        tangent = canonical_axis(fallback)
        nearest = polyline[0]
        progress = 0.
        total = 0.
    else:
        segments = np.diff(polyline, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        starts = np.r_[0., np.cumsum(lengths[:-1])]
        best = None
        for i, (a, d, length) in enumerate(zip(polyline[:-1], segments, lengths)):
            if length < EPS: continue
            t = np.clip(float((p-a) @ d / length**2), 0., 1.)
            near = a+t*d
            dist = float(np.linalg.norm(p-near))
            if best is None or dist < best[0]: best = (dist, near, canonical_axis(d), starts[i]+t*length)
        if best is None: tangent = canonical_axis(fallback); nearest = polyline[0]; progress = 0.
        else: _, nearest, tangent, progress = best
        total = float(np.sum(lengths))
    if tangent is None: tangent = np.array([1., 0.])
    normal = perpendicular(tangent)
    side = float((p-nearest) @ normal)
    point_geom = Point(float(p[0]), float(p[1]))
    distances = [float(point_geom.distance(poly)) for m in members if (poly := polygon_of(m.footprint)) is not None]
    distance = min(distances) if distances else float(np.linalg.norm(p-nearest))
    return RoadProjection(distance, np.asarray(nearest), tangent, normal, side,
                          progress, min(progress, max(0., total-progress)), distance <= buffer_m)


def road_crosses(a, b, members):
    from shapely.geometry import LineString
    from .edge_features import polygon_of
    line = LineString([tuple(np.asarray(a)[:2]), tuple(np.asarray(b)[:2])])
    return any(line.intersects(poly) for m in members if (poly := polygon_of(m.footprint)) is not None)


def between_features(target, anchor_a, anchor_b):
    t = np.asarray(target, dtype=float)[:2]; a = np.asarray(anchor_a, dtype=float)[:2]; b = np.asarray(anchor_b, dtype=float)[:2]
    d = b-a; span = float(np.linalg.norm(d)); da = float(np.linalg.norm(t-a)); db = float(np.linalg.norm(t-b))
    if span < EPS: return None
    ratio = float((t-a) @ d / span**2)
    projected = a+np.clip(ratio,0.,1.)*d
    perp = float(np.linalg.norm(t-projected))
    return {'between_t': ratio, 'between_perpendicular': perp,
            'between_distance_a': da, 'between_distance_b': db,
            'between_balance': abs(da-db)/(da+db+EPS),
            'between_anchor_separation': span,
            'between_inside_segment': float(0. < ratio < 1.)}


def frame_axes(row, anchor=None, road=None):
    """No target position is read while constructing a reference frame."""
    axes = {'GLOBAL': np.array([0.,1.]),
            'START': unit(row.get('start_heading')) if row.get('start_heading') is not None else None,
            'ROUTE_END': unit(row.get('route_heading')) if row.get('route_heading') is not None else None,
            'ROUTE_PCA': unit(row.get('route_pca_heading')) if row.get('route_pca_heading') is not None else None,
            'ROAD': road.tangent if road is not None else None}
    for n in WINDOWS:
        v=row.get(f'final_heading_{n}')
        axes[f'FINAL{n}']=unit(v) if v is not None else None
    major, minor = anchor_axes(anchor.footprint) if anchor is not None else (None,None)
    axes['ANCHOR_MAJOR']=major;axes['ANCHOR_MINOR']=minor
    return axes
