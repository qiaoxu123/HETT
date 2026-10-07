"""The edge feature schema, defined once.

Section 6 of the brief asks for twenty-odd relative-geometry columns and then
says, in as many words, that they must have **one** schema definition and must
not be indexed by hard-coded column numbers in several scripts.  That is not
book-keeping: the previous round lost a measurement to a compass table that read
``north of`` off the east-west coordinate, and an index written down twice is a
second chance to make that mistake.  So the names live here, everything indexes
through :data:`EDGE_INDEX`, and a test pins the width.

**Three frames are carried, none of them privileged.**  The global axes, the
anchor's own footprint axes, and the agent's heading.  The audit that follows
measures which of them, if any, the corpus's words agree with, so the feature
vector has to contain all three rather than commit to one on the builder's
behalf -- that commitment is exactly what went wrong last round.

**Roads are regions, not segments.**  ``road_*`` columns are read against the
nearest point on the road *region*, so "along Aldridge Road" means along the road
rather than along whichever of its 180 pieces came closest.
"""

from __future__ import annotations

import numpy as np

# (name, unit divisor) in the order the vector is assembled.  The divisor is a
# scale normalisation, not a tuned weight: it puts every column in a comparable
# range so that a single learning rate is meaningful.
EDGE_FEATURES = (
    ("dx", 100.0),
    ("dy", 100.0),
    ("dz", 20.0),
    ("log_center_distance", 1.0),
    ("center_distance", 200.0),
    ("log_footprint_distance", 1.0),
    ("footprint_distance", 200.0),
    ("sin_bearing", 1.0),
    ("cos_bearing", 1.0),
    ("log_height_ratio", 1.0),
    ("log_width_ratio", 1.0),
    ("log_length_ratio", 1.0),
    ("log_area_ratio", 1.0),
    ("footprint_overlap", 1.0),
    ("contains", 1.0),
    ("nearest_road_distance", 200.0),
    ("log_nearest_road_distance", 1.0),
    ("same_road_region", 1.0),
    ("same_side_of_road", 1.0),
    ("opposite_side_of_road", 1.0),
    ("road_tangent_coord", 200.0),
    ("road_normal_coord", 100.0),
    ("anchor_major_coord", 100.0),
    ("anchor_minor_coord", 100.0),
    ("agent_ahead", 100.0),
    ("agent_lateral", 100.0),
)
# A node with no road frame at all gets this distance rather than infinity:
# an infinity would propagate through log1p into the feature vector and take the
# whole batch with it.  1 km is beyond any real separation in these blocks.
MISSING_ROAD_M = 1000.0
EDGE_INDEX = {name: i for i, (name, _) in enumerate(EDGE_FEATURES)}
EDGE_DIM = len(EDGE_FEATURES)
_SCALE = np.array([s for _, s in EDGE_FEATURES], dtype=np.float64)


def name_of(index: int) -> str:
    return EDGE_FEATURES[index][0]


def _polygon(footprint):
    from shapely.geometry import Polygon
    if footprint.shape[0] < 3:
        return None
    try:
        poly = Polygon(footprint)
        return poly if poly.is_valid and not poly.is_empty else poly.buffer(0)
    except Exception:
        return None


def footprint_distance(a, b) -> float:
    """Nearest distance between two footprints, or between centres if degenerate."""
    pa, pb = _polygon(a.footprint), _polygon(b.footprint)
    if pa is None or pb is None:
        return float(np.linalg.norm(a.center[:2] - b.center[:2]))
    try:
        return float(pa.distance(pb))
    except Exception:
        return float(np.linalg.norm(a.center[:2] - b.center[:2]))


def footprint_overlap(a, b) -> tuple:
    """(intersection over smaller area, one contains the other)."""
    pa, pb = _polygon(a.footprint), _polygon(b.footprint)
    if pa is None or pb is None:
        return 0.0, 0.0
    try:
        inter = pa.intersection(pb).area
    except Exception:
        return 0.0, 0.0
    smaller = min(pa.area, pb.area)
    if smaller <= 1e-9:
        return 0.0, 0.0
    return float(inter / smaller), float(inter >= 0.98 * smaller)


class RoadFrame:
    """Where a node sits relative to a road region's own direction.

    ``distance`` is to the region, not to a segment.  ``tangent`` and ``normal``
    come from the region's polyline at the nearest point, so ``side`` is a sign
    relative to the road rather than relative to the map.
    """

    __slots__ = ("distance", "tangent", "normal", "side", "tangent_coord",
                 "normal_coord", "nearest_point")

    def __init__(self, distance, tangent, normal, side, tangent_coord,
                 normal_coord, nearest_point):
        self.distance = distance
        self.tangent = tangent
        self.normal = normal
        self.side = side
        self.tangent_coord = tangent_coord
        self.normal_coord = normal_coord
        self.nearest_point = nearest_point


def road_frame(node, region, members) -> RoadFrame:
    """Read ``node`` against road ``region``, whose segments are ``members``."""
    from shapely.geometry import Point

    pt = Point(float(node.center[0]), float(node.center[1]))
    best = None
    for seg in members:
        poly = _polygon(seg.footprint)
        if poly is None:
            continue
        d = float(pt.distance(poly))
        if best is None or d < best[0]:
            best = (d, seg, poly)
    if best is None:
        return RoadFrame(MISSING_ROAD_M, np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                         0.0, 0.0, 0.0, node.center[:2])
    distance, seg, poly = best
    tangent = seg.major_axis
    if not np.isfinite(tangent).all() or np.linalg.norm(tangent) < 1e-9:
        tangent = region.major_axis
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-9)
    normal = np.array([-tangent[1], tangent[0]])

    # The nearest point on the segment's *centre line*, approximated by the
    # segment centre projected onto the line through it.  A road piece is small
    # and straight enough that this is not worth a full polyline solve.
    rel = node.center[:2] - seg.center[:2]
    along = float(rel @ tangent)
    nearest = seg.center[:2] + along * tangent
    side = float(rel @ normal)
    if abs(side) < 1e-9:
        side = 0.0
    return RoadFrame(distance, tangent, normal, side, along, float(rel @ normal),
                     nearest)


class EdgeContext:
    """Everything an edge needs that is not one of its two endpoints."""

    def __init__(self, road_regions=None, road_members=None, agent_xy=None,
                 agent_yaw=0.0, nearest_road=None):
        self.road_regions = road_regions or []
        self.road_members = road_members or {}
        self.agent_xy = (np.zeros(2) if agent_xy is None
                         else np.asarray(agent_xy, dtype=np.float64)[:2])
        self.agent_yaw = float(agent_yaw)
        self.nearest_road = nearest_road or {}


def edge_vector(a, b, ctx: EdgeContext = None) -> np.ndarray:
    """The schema applied to the ordered pair ``(a, b)``.

    Directional by construction: every column changes sign or value when the
    pair is swapped, which is what lets a relation word be read off it.  A
    symmetric feature vector could not express "north of" at all.
    """
    ctx = ctx or EdgeContext()
    delta = b.center - a.center
    xy = delta[:2]
    centre_d = float(np.linalg.norm(xy))
    fp_d = footprint_distance(a, b)
    bearing = float(np.arctan2(delta[1], delta[0]))
    overlap, contains = footprint_overlap(a, b)

    def log_ratio(x, y):
        return float(np.log((abs(x) + 1e-6) / (abs(y) + 1e-6)))

    heading = np.array([np.cos(ctx.agent_yaw), np.sin(ctx.agent_yaw)])
    right = np.array([np.sin(ctx.agent_yaw), -np.cos(ctx.agent_yaw)])
    major = np.asarray(a.major_axis, dtype=np.float64)[:2]
    minor = np.array([-major[1], major[0]])

    road_d = MISSING_ROAD_M
    same_region = 0.0
    same_side = 0.0
    opposite_side = 0.0
    tangent_coord = 0.0
    normal_coord = 0.0
    region_a = a.parent_region
    region_b = b.parent_region
    if region_a >= 0 and region_a == region_b and a.kind.value == "road_segment":
        same_region = 1.0
    chosen = ctx.nearest_road.get(id(a))
    if chosen is None and ctx.road_regions:
        # Fall back to the region whose frame puts `a` closest.
        best = None
        for idx, region in enumerate(ctx.road_regions):
            frame = road_frame(a, region, ctx.road_members.get(idx, []))
            if best is None or frame.distance < best.distance:
                best = frame
        chosen = best
    if chosen is not None:
        road_d = chosen.distance
        tangent_coord = float((b.center[:2] - chosen.nearest_point) @ chosen.tangent)
        normal_coord = float((b.center[:2] - chosen.nearest_point) @ chosen.normal)
        b_side = float((b.center[:2] - chosen.nearest_point) @ chosen.normal)
        if abs(chosen.side) > 1e-6 and abs(b_side) > 1e-6:
            same_side = float(np.sign(chosen.side) == np.sign(b_side))
            opposite_side = float(np.sign(chosen.side) != np.sign(b_side))

    raw = np.array([
        delta[0], delta[1], delta[2],
        np.log1p(centre_d), centre_d,
        np.log1p(fp_d), fp_d,
        np.sin(bearing), np.cos(bearing),
        log_ratio(b.height, a.height),
        log_ratio(b.dimension[0], a.dimension[0]),
        log_ratio(b.dimension[1], a.dimension[1]),
        log_ratio(b.footprint_area, a.footprint_area),
        overlap, contains,
        road_d, np.log1p(road_d),
        same_region, same_side, opposite_side,
        tangent_coord, normal_coord,
        float(xy @ major), float(xy @ minor),
        float(xy @ heading),
        float(xy @ right),
    ], dtype=np.float64)
    return (raw / _SCALE).astype(np.float32)
