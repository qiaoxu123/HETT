"""What each relation means, geometrically, with no language in the loop.

The previous round's failure was traced to a lexicon that assigned a frame and a
sign to each word by hand and never checked them. This module is the other way
round: a relation is *defined* by a geometric predicate over the map, and the
question of whether any English word means it is deferred to Phase 4. That
ordering is the point of the round -- build a world model that is certainly
correct, then ask whether language can be mapped onto it, rather than trying to
recover geometry from words whose usage turns out to be inconsistent.

Two deliberate restrictions, both from §9 of the brief:

* **No viewpoint-dependent relations.**  ``behind``, ``in front of``, ``left of``
  and ``right of`` are absent.  Each needs a reference frame, the previous
  round's audit found no frame the corpus agrees on, and admitting them here
  would smuggle the failure back in.
* **No contact or support relations.**  ``on`` is absent for the same reason: it
  is the corpus's most frequent relation word and the audit could not pin it to
  any geometry.

Each relation returns a **signed margin** in metres, positive when it holds, so
that a threshold can be swept without redefining the geometry, and so that
"barely holds" and "clearly holds" are distinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RelationContext:
    """What a relation may need beyond its two endpoints.

    ``road`` is the region the relation is read against -- for ``across_road``
    the one between the two entities, for the side relations whichever region
    the caller has decided the sentence is about.  It is passed in rather than
    searched for here, because choosing the road is a language question and this
    module is not allowed to guess at one.
    """

    road: object = None
    road_members: tuple = ()
    intersections: tuple = field(default_factory=tuple)

RELATION_NAMES = (
    "north_of", "south_of", "east_of", "west_of",
    "near", "far",
    "between",
    "same_side_of_road", "opposite_side_of_road", "across_road",
    "along_road", "near_intersection",
    "aligned_with", "parallel_to", "perpendicular_to",
)

# Relations whose opposite is another relation in the ontology, used by the
# counterfactual arm of the gate.  The rest have no opposite and are tested
# against a wrong-but-applicable relation instead.
OPPOSITE = {
    "north_of": "south_of", "south_of": "north_of",
    "east_of": "west_of", "west_of": "east_of",
    "near": "far", "far": "near",
    "same_side_of_road": "opposite_side_of_road",
    "opposite_side_of_road": "same_side_of_road",
}

# Which relations need a second anchor to be defined at all.
NEEDS_SECOND_ANCHOR = frozenset({"between"})
# Which relations are only meaningful for a pair involving a road region.
ROAD_RELATIONS = frozenset({"same_side_of_road", "opposite_side_of_road",
                            "across_road", "along_road"})

# Default thresholds, in metres except where noted.  Section 11 of the brief
# forbids inventing these: the candidate ranges come from train_seen statistics
# and the chosen values are selected on val_seen, which
# ``synthetic_relations.select_thresholds`` does.  These are the fallbacks used
# only when no selection has been run.
DEFAULT_THRESHOLDS = {
    "cardinal_margin": 8.0,     # metres north/east before "north of" holds
    "near_m": 25.0,
    "far_m": 150.0,
    "between_perp_m": 20.0,
    "between_t_margin": 0.10,   # keep clear of the segment ends
    "along_cos": 0.94,
    "along_min_m": 15.0,
    "intersection_m": 20.0,
    "align_cos": 0.94,
    "perp_cos": 0.20,
    "side_min_m": 2.0,          # how far off the road counts as "on a side"
}

# The values offered to the val_seen selection, per threshold.
THRESHOLD_GRID = {
    "cardinal_margin": (4.0, 8.0, 15.0, 25.0),
    "near_m": (15.0, 25.0, 40.0, 60.0),
    "far_m": (100.0, 150.0, 250.0),
    "between_perp_m": (10.0, 20.0, 35.0),
    "between_t_margin": (0.05, 0.10, 0.20),
    "along_cos": (0.90, 0.94, 0.97),
    "intersection_m": (12.0, 20.0, 35.0),
    "align_cos": (0.90, 0.94, 0.97),
    "perp_cos": (0.10, 0.20, 0.35),
}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def between_frame(anchor, second, target) -> tuple:
    """Projection ratio and perpendicular offset of ``target`` on A->B."""
    a = np.asarray(anchor.center[:2], dtype=np.float64)
    b = np.asarray(second.center[:2], dtype=np.float64)
    c = np.asarray(target.center[:2], dtype=np.float64)
    span = b - a
    length = float(np.linalg.norm(span))
    if length < 1e-6:
        return 0.0, float("inf"), 0.0
    t = float((c - a) @ span) / (length ** 2)
    perp = float(np.linalg.norm((c - a) - t * span))
    return t, perp, length


def road_side(target, region, members, th) -> tuple:
    """Signed offset of ``target`` from the road region's centre line, in metres."""
    from .edge_features import road_frame
    frame = road_frame(target, region, members)
    return frame.side, frame.distance, frame.tangent, frame.nearest_point


def crosses(anchor, target, region, members) -> bool:
    """Does the straight line anchor -> target cross this road region?"""
    from shapely.geometry import LineString
    from .edge_features import polygon_of
    try:
        line = LineString([tuple(np.asarray(anchor.center[:2])),
                           tuple(np.asarray(target.center[:2]))])
    except Exception:
        return False
    for seg in members:
        poly = polygon_of(seg.footprint)
        if poly is not None and line.intersects(poly):
            return True
    return False


def satisfaction(relation: str, anchor, target, ctx, th=None,
                 second=None) -> float:
    """Signed margin in metres: positive means the relation holds.

    Returning a margin rather than a bool is what lets the threshold sweep in
    ``select_thresholds`` re-use one geometry pass, and what lets the hard
    negatives be chosen by *how nearly* they satisfy a relation rather than by
    a coin flip.
    """
    th = th or DEFAULT_THRESHOLDS
    delta = np.asarray(target.center, dtype=np.float64) - \
        np.asarray(anchor.center, dtype=np.float64)

    if relation == "north_of":
        return float(delta[1]) - th["cardinal_margin"]
    if relation == "south_of":
        return -float(delta[1]) - th["cardinal_margin"]
    if relation == "east_of":
        return float(delta[0]) - th["cardinal_margin"]
    if relation == "west_of":
        return -float(delta[0]) - th["cardinal_margin"]

    from .edge_features import footprint_distance
    d = footprint_distance(anchor, target)
    if relation == "near":
        return th["near_m"] - d
    if relation == "far":
        return d - th["far_m"]

    if relation == "between":
        if second is None:
            return float("-inf")
        t, perp, length = between_frame(anchor, second, target)
        margin = th["between_t_margin"]
        inside = min(t - margin, (1.0 - margin) - t)
        return float(min(inside * length, th["between_perp_m"] - perp))

    if relation in ROAD_RELATIONS:
        region, members = ctx.road, ctx.road_members
        if region is None:
            return float("-inf")
        side_a = road_side(anchor, region, members, th)[0]
        side_b = road_side(target, region, members, th)[0]
        if relation == "same_side_of_road":
            if abs(side_a) < th["side_min_m"] or abs(side_b) < th["side_min_m"]:
                return float("-inf")
            return float(np.sign(side_a) * np.sign(side_b) *
                         min(abs(side_a), abs(side_b)))
        if relation == "opposite_side_of_road":
            if abs(side_a) < th["side_min_m"] or abs(side_b) < th["side_min_m"]:
                return float("-inf")
            return float(-np.sign(side_a) * np.sign(side_b) *
                         min(abs(side_a), abs(side_b)))
        if relation == "across_road":
            from .edge_features import road_frame as _rf
            frame = _rf(target, region, members)
            crossed = crosses(anchor, target, region, members)
            if not crossed:
                return float("-inf")
            return float(frame.distance)
        if relation == "along_road":
            frame = _rf(target, region, members)
            tangent = frame.tangent
            direction = _unit(np.asarray(target.center[:2]) -
                              np.asarray(anchor.center[:2]))
            if np.linalg.norm(direction) < 1e-9:
                return float("-inf")
            cos = float(direction @ tangent)
            length = float(np.linalg.norm(np.asarray(target.center[:2]) -
                                          np.asarray(anchor.center[:2])))
            if abs(cos) < th["along_cos"] or length < th["along_min_m"]:
                return float("-inf")
            return float(abs(cos) * length - frame.distance)

    if relation == "near_intersection":
        best = None
        for node in ctx.intersections:
            d = float(np.linalg.norm(np.asarray(target.center[:2]) -
                                     np.asarray(node.center[:2])))
            best = d if best is None else min(best, d)
        if best is None:
            return float("-inf")
        return th["intersection_m"] - best

    axis_a = _unit(anchor.major_axis)
    axis_b = _unit(target.major_axis)
    direction = _unit(np.asarray(target.center[:2]) -
                      np.asarray(anchor.center[:2]))
    if relation == "aligned_with":
        if np.linalg.norm(direction) < 1e-9:
            return float("-inf")
        return float(abs(float(direction @ axis_a)) - th["align_cos"])
    if relation == "parallel_to":
        return float(abs(float(axis_a @ axis_b)) - th["align_cos"])
    if relation == "perpendicular_to":
        return float(th["perp_cos"] - abs(float(axis_a @ axis_b)))
    raise KeyError(relation)


def holds(relation: str, anchor, target, ctx, th=None, second=None) -> bool:
    return satisfaction(relation, anchor, target, ctx, th, second) > 0.0
