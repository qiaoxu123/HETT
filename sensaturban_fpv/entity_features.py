"""Structured, non-neural descriptions of one candidate entity.

The round this belongs to asks a diagnostic question: once the target region is
masked correctly, what information is the model still missing?  Answering it
means being able to hand a probe one *kind* of evidence at a time, so this module
produces four families and keeps them apart:

``appearance``
    Colour and texture of the entity's own measured points and of its top-down
    pixels.  Independent of any learned embedding.
``geometry``
    Size, shape, height, orientation and how it projects -- all from the
    annotation and the entity's own points.
``relation``
    Where the entity stands relative to the agent and to the rest of the map:
    distance and bearing to the UAV, distance to the nearest road surface,
    neighbourhood composition, distance to the nearest same-class entity.
``semantic``
    Object type and map position.

The split between *deployable* and *oracle* is enforced at the call site, not
here: every function takes what it is given and nothing reaches into an
annotation for the answer.  The one place an oracle enters is
:func:`anchor_relation_features`, which is handed an anchor chosen elsewhere and
is documented as needing one.
"""

from __future__ import annotations

import numpy as np

# SensatUrban's class ids, used to find road surface near an entity.
ROAD_CLASS = 7
CLASS_NAMES = ("Ground", "HighVegetation", "Building", "Wall", "Bridge", "Parking",
               "Rail", "TrafficRoad", "StreetFurniture", "Car", "Footpath", "Bike",
               "Water")
N_CLASSES = len(CLASS_NAMES)
OWN_CLASSES = set(CLASS_NAMES)

FEATURE_NAMES = {}


def _named(family: str, names):
    FEATURE_NAMES[family] = list(names)
    return len(names)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB (0..255) to hue/saturation/value, all in 0..1.

    Written out rather than pulled from a library so the tests can pin the hue
    convention (red at 0, wrapping) independently of any dependency's version.
    """
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    diff = mx - mn
    hue = np.zeros_like(mx)
    nz = diff > 1e-9
    idx = nz & (mx == r)
    hue[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6.0
    idx = nz & (mx == g) & (mx != r)
    hue[idx] = (b[idx] - r[idx]) / diff[idx] + 2.0
    idx = nz & (mx == b) & (mx != r) & (mx != g)
    hue[idx] = (r[idx] - g[idx]) / diff[idx] + 4.0
    hue = hue / 6.0
    sat = np.where(mx > 1e-9, diff / np.maximum(mx, 1e-9), 0.0)
    return np.stack([hue, sat, mx], axis=1)


def colour_histogram(rgb: np.ndarray, bins: int = 6) -> np.ndarray:
    """Joint-normalised marginal histograms, one per channel."""
    if len(rgb) == 0:
        return np.zeros(3 * bins, dtype=np.float32)
    out = []
    for channel in range(3):
        counts, _ = np.histogram(rgb[:, channel], bins=bins, range=(0, 255))
        out.append(counts.astype(np.float64) / max(len(rgb), 1))
    return np.concatenate(out).astype(np.float32)


def colour_entropy(rgb: np.ndarray, bins: int = 8) -> float:
    """Shannon entropy of the joint colour histogram, in nats."""
    if len(rgb) == 0:
        return 0.0
    q = np.clip((rgb / 256.0 * bins).astype(np.int64), 0, bins - 1)
    flat = q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2]
    counts = np.bincount(flat, minlength=bins ** 3).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def appearance_features(rgb_points: np.ndarray, patch: np.ndarray | None = None,
                        grey: np.ndarray | None = None) -> np.ndarray:
    """Colour and texture of the entity.

    ``rgb_points`` are the entity's own measured point colours -- the appearance
    the dataset actually recorded, before any render.  ``patch``, when given, is
    the top-down raster crop the entity's mask selected, and contributes
    image-domain statistics: brightness spread, edge density and local contrast.
    A number of these features are cheap and correlated on purpose; the probe is
    linear and regularised, and dropping correlated columns by hand would be
    choosing the answer.
    """
    rgb = np.asarray(rgb_points, dtype=np.float64).reshape(-1, 3)
    feats = []
    if len(rgb):
        hsv = rgb_to_hsv(rgb)
        feats += [rgb.mean(axis=0), np.median(rgb, axis=0), rgb.std(axis=0),
                  hsv.mean(axis=0), hsv.std(axis=0)]
        feats.append(colour_histogram(rgb))
        feats.append(np.array([
            colour_entropy(rgb),
            float(np.mean(rgb.std(axis=1) < 12)),        # how grey it is
            float(np.mean(rgb.mean(axis=1) < 60)),       # how dark it is
            float(np.mean(rgb.mean(axis=1) > 200)),      # how bright it is
            float(np.log1p(len(rgb))),
        ]))
        feats.append(np.array([float(np.mean(hsv[:, 1] > 0.25)),  # saturated share
                               float(np.std(hsv[:, 0]))], dtype=np.float64))
    else:
        feats += [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
                  np.zeros(18), np.zeros(5), np.zeros(2)]

    if patch is not None and np.asarray(patch).size:
        img = np.asarray(patch, dtype=np.float64)
        flat = img.reshape(-1, 3) if img.ndim == 3 else img.reshape(-1, 1)
        grey_img = (flat @ np.array([0.299, 0.587, 0.114])
                    if flat.shape[1] == 3 else flat[:, 0])
        if grey is not None:
            grey_img = np.asarray(grey_img)[np.asarray(grey).reshape(-1)]
        if grey_img.size:
            gx = np.diff(grey_img) if grey_img.size > 1 else np.zeros(1)
            feats.append(np.array([
                float(grey_img.mean()), float(grey_img.std()),
                float(np.percentile(grey_img, 10)),
                float(np.percentile(grey_img, 90)),
                float(np.mean(np.abs(gx) > 12.0)),        # edge density
            ]))
        else:
            feats.append(np.zeros(5))
    else:
        feats.append(np.zeros(5))

    out = np.concatenate([np.asarray(f, dtype=np.float64).reshape(-1) for f in feats])
    return out.astype(np.float32)


def geometry_features(points: np.ndarray, position, dimension, uav_position,
                      uav_yaw: float, projected_px: float = 0.0,
                      patch_count: float = 0.0) -> np.ndarray:
    """Size, shape, height, orientation and projection of one entity.

    The measure that is easiest to get wrong for small objects is the ratio
    between annotation and measurement: a car's annotated box and its measured
    point cloud disagree by more than a building's, and the disagreement is
    itself informative, so both are reported rather than one being chosen.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    dim = np.asarray(dimension, dtype=np.float64).reshape(3)
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    uav = np.asarray(uav_position, dtype=np.float64).reshape(3)

    if len(pts):
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = hi - lo
        z = pts[:, 2]
        z_mean, z_std = float(z.mean()), float(z.std())
        z_range = float(np.percentile(z, 95) - np.percentile(z, 5))
        xy = pts[:, :2] - pts[:, :2].mean(axis=0)
        cov = xy.T @ xy / max(len(xy), 1)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, int(np.argmax(evals))]
        angle = float(np.arctan2(major[1], major[0]))
        elong = float(np.sqrt(max(evals.max(), 0) / max(evals.min(), 1e-9)))
    else:
        span = np.zeros(3)
        z_mean = z_std = z_range = 0.0
        angle, elong = 0.0, 1.0

    footprint = float(span[0] * span[1])
    volume = float(span[0] * span[1] * max(span[2], 1e-3))
    density = float(len(pts) / max(footprint, 1e-3))
    compactness = float(len(pts) / max(volume, 1e-3))
    small = max(min(dim[0], dim[1]), 1e-3)
    delta = pos - uav
    distance = float(np.linalg.norm(delta))
    bearing = float(np.arctan2(delta[1], delta[0]) - uav_yaw)
    rel = np.array([np.cos(bearing), np.sin(bearing)])

    return np.array([
        float(dim[0]), float(dim[1]), float(dim[2]),
        float(np.log1p(max(dim[0] * dim[1] * dim[2], 0.0))),
        float(max(dim[0], dim[1]) / small),                # aspect ratio
        float(span[0]), float(span[1]), float(span[2]),
        float(np.log1p(max(footprint, 0.0))),
        float(np.log1p(max(volume, 0.0))),
        float(np.log1p(len(pts))),
        float(np.log1p(max(density, 0.0))),
        float(np.log1p(max(compactness, 0.0))),
        z_mean, z_std, z_range,
        float(np.sin(2 * angle)), float(np.cos(2 * angle)),
        float(np.log1p(elong)),
        float(np.log1p(max(projected_px, 0.0))),
        float(np.log1p(max(patch_count, 0.0))),
        float(np.log1p(max(distance, 0.0))), float(distance / 400.0),
        rel[0], rel[1],                                    # relative bearing
        float(np.log1p(max(span[0] * span[1] / max(dim[0] * dim[1], 1e-3), 0.0))),
    ], dtype=np.float32)


def _nearest_distance(target_xy: np.ndarray, others_xy: np.ndarray) -> float:
    if len(others_xy) == 0:
        return -1.0
    d = np.linalg.norm(others_xy - target_xy[None, :], axis=1)
    return float(d.min())


def relation_features(position, uav_position, uav_yaw: float,
                      others_position: np.ndarray, others_type: np.ndarray,
                      own_type: str, road: tuple | None = None,
                      max_radius: float = 80.0,
                      others_class: np.ndarray | None = None) -> np.ndarray:
    """Where this candidate stands, relative to the agent and to the map.

    Everything here is computed from the candidate list, the annotation and the
    agent's pose -- all of which exist at test time.  Nothing consults which
    candidate is the target.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    uav = np.asarray(uav_position, dtype=np.float64).reshape(3)
    xy = pos[:2]
    delta = pos - uav
    distance = float(np.linalg.norm(delta))
    # Bearing relative to the agent's heading, as a direction on the image plane:
    # positive is to the right of the heading, which is what "to the left of the
    # road" and "behind the van" are expressed in.
    heading = np.array([np.cos(uav_yaw), np.sin(uav_yaw)])
    right = np.array([np.sin(uav_yaw), -np.cos(uav_yaw)])
    ahead = float(delta[:2] @ heading)
    lateral = float(delta[:2] @ right)

    others = np.asarray(others_position, dtype=np.float64).reshape(-1, 3)
    types = np.asarray(others_type).reshape(-1)
    dists = (np.linalg.norm(others[:, :2] - xy[None, :], axis=1)
             if len(others) else np.zeros(0))
    # The neighbourhood is the whole block -- hundreds to thousands of entities
    # -- so the class histogram is built with one bincount over precomputed
    # class indices rather than a Python loop, which dominated the pass.
    # Named class indices for the whole block, computed once per map by the
    # caller: the neighbourhood is every entity in the block, and rebuilding a
    # 2000-element list per candidate dominated the pass.
    class_index = (others_class if others_class is not None else
                   np.array([CLASS_NAMES.index(t) if t in CLASS_NAMES else -1
                             for t in types.tolist()], dtype=np.int64))
    same = types == own_type
    near = dists <= max_radius
    hist = np.zeros(N_CLASSES, dtype=np.float64)
    if len(others):
        visible = class_index[near]
        visible = visible[visible >= 0]
        if visible.size:
            hist = np.bincount(visible, minlength=N_CLASSES).astype(np.float64)
            hist /= hist.sum()
        own_type_index = CLASS_NAMES.index(own_type) if own_type in CLASS_NAMES else -1
        if 0 <= own_type_index < N_CLASSES:
            hist[own_type_index] = 0.0  # own class share is reported separately
    own_share = float(np.mean(same[near])) if len(others) and near.any() else 0.0

    if road is not None:
        road_distance, road_frac = float(road[0]), float(road[1])
        road_min = float(np.log1p(road_distance)) if road_distance >= 0 else -1.0
    else:
        road_min, road_frac = -1.0, 0.0

    return np.concatenate([np.array([
        np.log1p(max(distance, 0.0)), distance / 400.0,
        float(np.log1p(abs(ahead))), float(np.sign(ahead)),
        float(np.log1p(abs(lateral))), float(np.sign(lateral)),
        float(abs(np.arctan2(lateral, ahead))),             # off-axis angle
        float(np.log1p(max(dists.min(), 0.0))) if len(dists) else 0.0,
        np.log1p(max(_nearest_distance(xy, others[~same][:, :2])
                     if len(others[~same]) else 0.0, 0.0)),
        np.log1p(max(_nearest_distance(xy, others[same][:, :2])
                     if len(others[same]) else 0.0, 0.0)),
        float(np.mean(near)) if len(dists) else 0.0,
        float(np.sum(dists <= 30.0)) / 10.0,
        own_share,
        road_min, road_frac,
    ]), hist]).astype(np.float32)


def semantic_features(object_type: str, position, block_bounds, entity_name: str = "",
                      n_entities_in_map: int = 1, id_bucket: int = -1) -> np.ndarray:
    """Object type and where the entity sits in its block.

    ``block_bounds`` is ``(lo, hi)`` of the point cloud in world XY.  The
    normalised coordinate is a *memorisation diagnostic*: a probe given it can
    learn "the target is always near this corner of this block", which works on
    the training maps and cannot work on unseen ones.  It is included so that
    the collapse is measured rather than assumed.
    """
    onehot = np.zeros(N_CLASSES + 1, dtype=np.float64)
    if object_type in CLASS_NAMES:
        onehot[CLASS_NAMES.index(object_type)] = 1.0
    else:
        onehot[-1] = 1.0
    lo, hi = np.asarray(block_bounds[0], float)[:2], np.asarray(block_bounds[1], float)[:2]
    span = np.maximum(hi - lo, 1e-6)
    norm = (np.asarray(position, float)[:2] - lo) / span
    return np.concatenate([onehot, norm, [
        float(np.log1p(max(n_entities_in_map, 0))) / 10.0,
        1.0 if entity_name else 0.0,
        float(id_bucket),
    ]]).astype(np.float32)


def anchor_relation_features(position, anchor_position, anchor_type: str,
                             own_type: str, uav_position, uav_yaw: float) -> np.ndarray:
    """Relation to a *named anchor*, for the oracle arm only.

    Choosing the anchor needs to know which entity the language refers to, which
    is exactly what the task is about; the caller supplies one and must label the
    result ORACLE.  The features are the same shape as the plain relation block
    so that a probe can be swapped between the two arms without changing shape.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    anc = np.asarray(anchor_position, dtype=np.float64).reshape(3)
    uav = np.asarray(uav_position, dtype=np.float64).reshape(3)
    d = pos - anc
    distance = float(np.linalg.norm(d))
    heading = np.array([np.cos(uav_yaw), np.sin(uav_yaw)])
    right = np.array([np.sin(uav_yaw), -np.cos(uav_yaw)])
    rel = d[:2]
    return np.array([
        np.log1p(distance),
        1.0 / (1.0 + distance),
        float(np.exp(-distance / 20.0)),
        float(np.exp(-distance / 60.0)),
        float(rel @ heading), float(rel @ right),
        float(np.arctan2(rel[1], rel[0])),
        float(np.cos(np.arctan2(rel[1], rel[0]))),
        float(np.sin(np.arctan2(rel[1], rel[0]))),
        1.0 if anchor_type == own_type else 0.0,
        float(np.log1p(np.linalg.norm(pos - uav))),
    ], dtype=np.float32)


def standardise(train: np.ndarray, others: list) -> tuple:
    """Mean/std from the *training* split only, applied to every split.

    Fitting the scaler on all splits would leak the held-out distribution into
    every probe, which on a diagnostic whose whole point is transfer to unseen
    maps would quietly manufacture the result.
    """
    mean = np.asarray(train, dtype=np.float64).mean(axis=0)
    std = np.asarray(train, dtype=np.float64).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (mean.astype(np.float32), std.astype(np.float32),
            [((np.asarray(o, np.float64) - mean) / std).astype(np.float32)
             for o in others])


class RoadIndex:
    """Per-block road presence on the bucket grid, built once per block.

    Asking the point cloud "where is the nearest road surface" by reading a
    45 m box of points costs millions of points per candidate and dominated the
    feature pass.  The bucket index already groups points by cell and the label
    array already says which of them are road, so reducing the label mask over
    each cell's slot range gives a coarse road raster for the whole block in one
    pass -- and a per-candidate lookup then reads a few thousand booleans.
    """

    def __init__(self, grid, labels):
        starts = np.asarray(grid.starts)
        n_cells = len(starts) - 1
        counts = np.zeros(n_cells, dtype=np.int64)
        if labels is not None and len(labels):
            is_road = np.asarray(labels) == ROAD_CLASS
            nonempty = np.flatnonzero(starts[1:] > starts[:-1])
            if nonempty.size:
                # reduceat needs strictly increasing indices for the segments it
                # is given, so the empty cells are skipped rather than included
                # in the run.
                counts[nonempty] = np.add.reduceat(
                    is_road, np.maximum(starts[:-1][nonempty], 0))
        self.road = counts.reshape(grid.nx, grid.ny) > 0
        self.lo = np.asarray(grid.lo, dtype=np.float64)[:2]
        self.cell = float(grid.cell)

    def _cell_xy(self, x: float, y: float) -> tuple:
        return (int((x - self.lo[0]) / self.cell), int((y - self.lo[1]) / self.cell))

    def distance_m(self, x: float, y: float, radius_cells: int = 24) -> tuple:
        """Distance to the nearest road cell, and the share of road nearby."""
        cx, cy = self._cell_xy(x, y)
        x0, x1 = max(cx - radius_cells, 0), min(cx + radius_cells + 1, self.road.shape[0])
        y0, y1 = max(cy - radius_cells, 0), min(cy + radius_cells + 1, self.road.shape[1])
        if x0 >= x1 or y0 >= y1:
            return -1.0, 0.0
        window = self.road[x0:x1, y0:y1]
        if not window.any():
            return -1.0, 0.0
        ix, iy = np.nonzero(window)
        dx = (x0 + ix + 0.5) * self.cell + self.lo[0] - x
        dy = (y0 + iy + 0.5) * self.cell + self.lo[1] - y
        d = np.sqrt(dx * dx + dy * dy)
        return float(d.min()), float(np.mean(d <= 15.0))
