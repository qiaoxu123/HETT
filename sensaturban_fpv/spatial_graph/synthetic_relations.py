"""Clean supervision, generated from the map instead of from the corpus.

The round's ordering exists because the previous one failed for a reason that
was located precisely: the corpus's relation words do not select any geometry,
so training on them teaches nothing and the failure cannot be diagnosed from the
outside.  Here the labels come from the map, where the geometry *is* the label,
so a model that fails to fit them is failing at something demonstrably
learnable.

**The split is by map, not by CityNav split.**  ``val_seen`` shares all 23 of its
maps with ``train_seen`` -- only ``val_unseen``'s 4 maps are disjoint (measured,
not assumed).  For instruction-level work that is a defensible split; for
supervision derived from the map's own geometry it is not, because a held-out
*episode* on a map the model has already seen is not held out.  So the 30 maps
outside ``val_unseen`` are divided by map into synthetic-train and
synthetic-val, and ``val_unseen``'s 4 maps are touched once, at the end.

**Hard negatives are matched on everything except the relation.**  A negative
drawn at random lets a model separate positive from negative on distance, entity
kind or local density without ever reading the relation -- which is what the
previous round's reasoner did.  Each negative is drawn from the same kind group
and the same distance band as the positive, and differs only in failing the
relation.

**Thresholds are cheap.**  The geometry of a pair is computed once into raw
facts; a threshold is then arithmetic on those facts, so the grid sweep in
:func:`select_thresholds` costs no further map work.
"""

from __future__ import annotations

import numpy as np

from .node_types import NodeKind
from .relation_geometry import (
    DEFAULT_THRESHOLDS, NEEDS_SECOND_ANCHOR, OPPOSITE, RELATION_NAMES,
    THRESHOLD_GRID, between_frame, crosses,
)

TARGET_KINDS = (NodeKind.BUILDING, NodeKind.LANDMARK, NodeKind.OBJECT,
                NodeKind.ROAD_REGION)
ANCHOR_KINDS = (NodeKind.BUILDING, NodeKind.LANDMARK, NodeKind.ROAD_REGION)

NEIGHBOUR_RADIUS_M = 250.0
MAX_TARGETS_PER_ANCHOR = 40
MAX_NEGATIVES = 8
DISTANCE_BAND = (0.5, 2.0)
MIN_EXAMPLES_PER_RELATION = 20

# Relations whose value depends on a threshold; the rest are fixed by geometry.
THRESHOLD_DRIVEN = {
    "north_of": "cardinal_margin", "south_of": "cardinal_margin",
    "east_of": "cardinal_margin", "west_of": "cardinal_margin",
    "near": "near_m", "far": "far_m",
    "between": "between_perp_m",
    "along_road": "along_cos", "near_intersection": "intersection_m",
    "aligned_with": "align_cos", "parallel_to": "align_cos",
    "perpendicular_to": "perp_cos",
    "same_side_of_road": "side_min_m",
    "opposite_side_of_road": "side_min_m",
}


class Grid:
    """Bucket a block's nodes so neighbour lookup is not a full sweep."""

    def __init__(self, nodes, cell: float = 50.0):
        self.cell = cell
        self.buckets = {}
        for node in nodes:
            key = (int(node.center[0] // cell), int(node.center[1] // cell))
            self.buckets.setdefault(key, []).append(node)

    def near(self, xy, radius: float) -> list:
        r = int(np.ceil(radius / self.cell))
        cx, cy = int(xy[0] // self.cell), int(xy[1] // self.cell)
        out = []
        for i in range(cx - r, cx + r + 1):
            for j in range(cy - r, cy + r + 1):
                out.extend(self.buckets.get((i, j), ()))
        return out


def _road_for(graph, node):
    """The road region a relation about ``node`` is read against.

    The nearest one, chosen geometrically.  Mapping "near Aldridge Road" to the
    region Aldridge Road is a *language* question and belongs to Phase 4; this
    is the map-side default, and the audit reports how often a node has two
    regions within the same distance band, which is where that default could
    mislead.
    """
    from .edge_features import road_frame
    best, best_d = None, None
    for idx, region in enumerate(graph.regions):
        members = graph.context.road_members.get(idx, [])
        if not members:
            continue
        frame = road_frame(node, region, members)
        if best_d is None or frame.distance < best_d:
            best, best_d = (region, tuple(members)), frame.distance
    return best if best is not None else (None, ())


def pair_facts(graph, anchor, target, road=None) -> dict:
    """Every raw quantity the ontology needs, computed once per pair."""
    from .edge_features import EdgeContext, edge_vector, footprint_distance, road_frame

    if road is None:
        road = _road_for(graph, anchor)
    region, members = road
    delta = np.asarray(target.center, dtype=np.float64) - \
        np.asarray(anchor.center, dtype=np.float64)
    facts = {
        "dx": float(delta[0]),
        "dy": float(delta[1]),
        "footprint_distance": footprint_distance(anchor, target),
        "centre_distance": float(np.linalg.norm(delta[:2])),
    }
    axis_a = anchor.major_axis / max(float(np.linalg.norm(anchor.major_axis)), 1e-9)
    axis_b = target.major_axis / max(float(np.linalg.norm(target.major_axis)), 1e-9)
    facts["parallel_cos"] = abs(float(axis_a @ axis_b))
    direction = delta[:2]
    norm = float(np.linalg.norm(direction))
    facts["aligned_cos"] = (abs(float((direction / norm) @ axis_a))
                            if norm > 1e-9 else 0.0)
    # Side of the anchor's road, and whether the anchor-to-target line crosses it.
    if region is not None and members:
        frame_a = road_frame(anchor, region, list(members))
        frame_b = road_frame(target, region, list(members))
        facts["side_a"], facts["side_b"] = frame_a.side, frame_b.side
        facts["road_distance"] = frame_b.distance
        facts["crosses_road"] = crosses(anchor, target, region, list(members))
        tangent = frame_b.tangent
        if norm > 1e-9:
            facts["along_cos"] = abs(float((direction / norm) @ tangent))
        else:
            facts["along_cos"] = 0.0
    else:
        facts["side_a"] = facts["side_b"] = 0.0
        facts["road_distance"] = 1000.0
        facts["crosses_road"] = False
        facts["along_cos"] = 0.0
    # Distance to the nearest intersection node, if the block has any.
    inter = graph.nodes_of(NodeKind.INTERSECTION)
    if inter:
        facts["intersection_distance"] = min(
            float(np.linalg.norm(np.asarray(target.center[:2]) -
                                 np.asarray(n.center[:2]))) for n in inter)
    else:
        facts["intersection_distance"] = 1000.0
    facts["edge"] = edge_vector(
        anchor, target,
        EdgeContext(road_regions=graph.regions,
                    road_members=graph.context.road_members,
                    cross_road=(region, members)))
    facts["road"] = road
    facts["second"] = None
    return facts


def margin_from_facts(relation: str, facts: dict, th: dict,
                      second_facts: dict = None) -> float:
    """The signed margin of ``relation``, as arithmetic on ``facts``."""
    if relation == "north_of":
        return facts["dy"] - th["cardinal_margin"]
    if relation == "south_of":
        return -facts["dy"] - th["cardinal_margin"]
    if relation == "east_of":
        return facts["dx"] - th["cardinal_margin"]
    if relation == "west_of":
        return -facts["dx"] - th["cardinal_margin"]
    if relation == "near":
        return th["near_m"] - facts["footprint_distance"]
    if relation == "far":
        return facts["footprint_distance"] - th["far_m"]
    if relation == "between":
        if second_facts is None:
            return float("-inf")
        t, perp, length = second_facts
        margin = th["between_t_margin"]
        inside = min(t - margin, (1.0 - margin) - t)
        return float(min(inside * length, th["between_perp_m"] - perp))
    if relation == "same_side_of_road":
        a, b = facts["side_a"], facts["side_b"]
        floor = th["side_min_m"]
        if abs(a) < floor or abs(b) < floor:
            return float("-inf")
        return float(np.sign(a) * np.sign(b) * min(abs(a), abs(b)))
    if relation == "opposite_side_of_road":
        a, b = facts["side_a"], facts["side_b"]
        floor = th["side_min_m"]
        if abs(a) < floor or abs(b) < floor:
            return float("-inf")
        return float(-np.sign(a) * np.sign(b) * min(abs(a), abs(b)))
    if relation == "across_road":
        if not facts["crosses_road"]:
            return float("-inf")
        return float(facts["road_distance"])
    if relation == "along_road":
        if facts["along_cos"] < th["along_cos"]:
            return float("-inf")
        return float(facts["along_cos"] * facts["centre_distance"]
                     - facts["road_distance"])
    if relation == "near_intersection":
        return th["intersection_m"] - facts["intersection_distance"]
    if relation == "aligned_with":
        return facts["aligned_cos"] - th["align_cos"]
    if relation == "parallel_to":
        return facts["parallel_cos"] - th["align_cos"]
    if relation == "perpendicular_to":
        return th["perp_cos"] - facts["parallel_cos"]
    raise KeyError(relation)


def build_pairs(graph, rng) -> dict:
    """All (anchor, target) pairs within range, with their raw facts, per anchor."""
    targets = [n for n in graph.nodes if n.kind in TARGET_KINDS]
    anchors = [n for n in graph.nodes if n.kind in ANCHOR_KINDS]
    if not targets or not anchors:
        return {}
    grid = Grid(targets)
    out = {}
    for anchor in anchors:
        road = _road_for(graph, anchor)
        seen = []
        for target in grid.near(anchor.center[:2], NEIGHBOUR_RADIUS_M):
            if target.node_id == anchor.node_id:
                continue
            d = float(np.linalg.norm(np.asarray(target.center[:2]) -
                                     np.asarray(anchor.center[:2])))
            if d <= NEIGHBOUR_RADIUS_M:
                seen.append((d, target))
        if not seen:
            continue
        # Stratified rather than nearest-first.  Taking the closest 40 leaves
        # the "far" relation with no positives at all, because everything within
        # 150 m fails it by construction -- a bias that would silently drop one
        # of the ontology's relations from the dataset.
        seen.sort(key=lambda x: x[0])
        half = MAX_TARGETS_PER_ANCHOR // 2
        nearest = seen[:half]
        rest = seen[half:]
        rng.shuffle(rest)
        selected = nearest + rest[:MAX_TARGETS_PER_ANCHOR - half]
        out[anchor.node_id] = {
            "anchor": anchor,
            "road": road,
            "targets": [(t, pair_facts(graph, anchor, t, road))
                        for _, t in selected],
        }
    return out


def _second_anchors(graph, anchor, target, pairs, rng, limit=3):
    """Candidate second anchors that make ``target`` lie between them."""
    out = []
    for other in graph.nodes:
        if other.kind not in ANCHOR_KINDS or other.node_id == anchor.node_id:
            continue
        t, perp, length = between_frame(anchor, other, target)
        if length < 20.0:
            continue
        out.append((t, perp, length, other))
    if not out:
        return []
    rng.shuffle(out)
    out.sort(key=lambda x: x[1])
    return out[:limit]


def build_block_examples(graph, thresholds, rng, split: str,
                         with_between: bool = True) -> list:
    """Every usable (anchor, relation, positive, matched negatives) in a block."""
    pairs = build_pairs(graph, rng)
    examples = []
    for anchor_id, bundle in pairs.items():
        anchor = bundle["anchor"]
        entries = []
        for target, facts in bundle["targets"]:
            margins = {r: margin_from_facts(r, facts, thresholds)
                       for r in RELATION_NAMES if r != "between"}
            entries.append((target, facts, margins))
        if not entries:
            continue
        if with_between:
            seen_targets = {t.node_id for t, _, _ in entries}
            for target, facts, margins in list(entries):
                for t, perp, length, other in _second_anchors(
                        graph, anchor, target, pairs, rng):
                    margins["between"] = margin_from_facts(
                        "between", facts, thresholds, (t, perp, length))
                    facts["second_id"] = int(other.node_id)
                    break
        for relation in RELATION_NAMES:
            positives = [e for e in entries if e[2].get(relation, -np.inf) > 0]
            pool = [e for e in entries
                    if relation in e[2] and e[2][relation] <= 0]
            if not positives or not pool:
                continue
            for target, facts, margins in positives[:MAX_TARGETS_PER_ANCHOR]:
                band = facts["centre_distance"]
                if band <= 1e-6:
                    continue
                group = target.kind.value
                hard = [(t, f, e) for t, f, e in pool
                        if t.kind.value == group
                        and DISTANCE_BAND[0] * band
                        <= f["centre_distance"] <= DISTANCE_BAND[1] * band]
                if not hard:
                    continue
                rng.shuffle(hard)
                chosen = hard[:MAX_NEGATIVES]
                examples.append({
                    "split": split,
                    "map": graph.map_name,
                    "relation": relation,
                    "anchor_id": int(anchor.node_id),
                    "anchor_kind": anchor.kind.value,
                    "positive_id": int(target.node_id),
                    "positive_edge": facts["edge"].tolist(),
                    "positive_margin": float(margins[relation]),
                    "negative_ids": [int(t.node_id) for t, _, _ in chosen],
                    "negative_edges": [f["edge"].tolist() for _, f, _ in chosen],
                    "opposite": OPPOSITE.get(relation),
                })
    return examples


def pack_facts(blocks, seed: int = 0) -> list:
    """Pull the raw quantities out of the cached pairs, once, as arrays.

    The threshold sweep evaluates tens of grid points over the val maps, and the
    first version recomputed every margin from the fact dictionaries on each
    pass -- roughly five million interpreted calls, which took longer than
    building the graphs.  Packing the facts into arrays once turns each sweep
    into vector arithmetic and makes section 11's requirement affordable.
    """
    packed = []
    for graph in blocks:
        pairs = build_pairs(graph, np.random.default_rng(seed))
        rows = []
        for bundle in pairs.values():
            for target, facts in bundle["targets"]:
                rows.append((
                    facts["dx"], facts["dy"], facts["footprint_distance"],
                    facts["centre_distance"], facts["side_a"], facts["side_b"],
                    1.0 if facts["crosses_road"] else 0.0,
                    facts["road_distance"], facts["along_cos"],
                    facts["intersection_distance"], facts["aligned_cos"],
                    facts["parallel_cos"], target.kind.value))
        if rows:
            packed.append(rows)
    return packed


def scale_grids(packed) -> dict:
    """Candidate threshold ranges taken from the maps' own scale.

    Section 11 forbids inventing the ranges.  These are quantiles of the
    centre-distance distribution *on the val maps*, so a threshold is expressed
    in units of how far apart things actually are in these blocks rather than in
    metres chosen from outside.
    """
    dists = np.concatenate([np.array([r[3] for r in rows]) for rows in packed
                            if rows]) if packed else np.array([1.0])
    q = lambda p: float(np.percentile(dists, p))
    return {
        "cardinal_margin": tuple(sorted({round(q(5)), round(q(10)), round(q(25)),
                                         round(q(50))})),
        "near_m": tuple(sorted({round(q(10)), round(q(25)), round(q(50)),
                                round(q(75))})),
        "far_m": tuple(sorted({round(q(75)), round(q(90)), round(q(97))})),
        "between_perp_m": tuple(sorted({round(q(5)), round(q(10)), round(q(25))})),
        "intersection_m": tuple(sorted({round(q(5)), round(q(10)), round(q(25))})),
        # The angular thresholds are not lengths and keep their unit grids.
        "between_t_margin": THRESHOLD_GRID["between_t_margin"],
        "along_cos": THRESHOLD_GRID["along_cos"],
        "align_cos": THRESHOLD_GRID["align_cos"],
        "perp_cos": THRESHOLD_GRID["perp_cos"],
    }


# How selective each relation should be: the fraction of nearby anchor-target
# pairs it should hold for.  See the note in ``select_thresholds`` for why a
# target rate is used rather than a maximum of some score.
TARGET_SELECTIVITY = {
    "cardinal_margin": 0.15,
    "near_m": 0.20,
    "far_m": 0.20,
    "between_perp_m": 0.10,
    "between_t_margin": 0.15,
    "along_cos": 0.15,
    "intersection_m": 0.10,
    "align_cos": 0.20,
    "perp_cos": 0.20,
    "side_min_m": 0.15,
}


def select_thresholds(blocks, base=None, seed: int = 0) -> dict:
    """Choose each threshold on the synthetic-val maps, one at a time.

    **Why a target rate, and not a score to maximise.**  Two natural criteria
    were tried and both are degenerate, which is worth recording because each
    looked reasonable:

    * maximising the *number* of usable examples walks every threshold to its
      loosest value -- loosening admits positives faster than it removes the
      negatives they are compared against, so the count is monotone, and the
      result is "everything is north of everything";
    * maximising the *total margin retained* walks every threshold to its
      tightest, since each surviving example then contributes more.

    Neither direction is a fact about the corpus; both are artefacts of the
    objective.  So the threshold is set to a **selectivity**: the value at which
    the relation holds for a stated fraction of nearby pairs.  That is a design
    decision and is reported as one rather than dressed up as something learned
    -- no data in this project says how far north "north of" starts.  What *is*
    data-determined is the grid it is chosen from, which ``scale_grids`` takes
    from the maps' own distance quantiles, and the rate is evaluated on the val
    maps rather than assumed.
    """
    packed = pack_facts(blocks, seed)
    grids = scale_grids(packed)
    chosen = dict(base or DEFAULT_THRESHOLDS)
    history = []
    for key, grid in grids.items():
        if key in ("between_perp_m", "between_t_margin"):
            # No single-anchor margin measures these; they are set from the
            # between-frame distribution in ``scale_grids`` and kept at the
            # tightest grid value, which is the most selective reading.
            chosen[key] = grid[0]
            history.append({"threshold": key, "chosen": grid[0],
                            "grid": list(grid), "target_rate": None,
                            "rate": None})
            continue
        target = TARGET_SELECTIVITY.get(key, 0.15)
        best, best_gap, best_rate = chosen[key], None, None
        for value in grid:
            rate = _positive_rate(packed, dict(chosen, **{key: value}), key)
            gap = abs(rate - target)
            if best_gap is None or gap < best_gap:
                best, best_gap, best_rate = value, gap, rate
        history.append({"threshold": key, "chosen": best, "grid": list(grid),
                        "target_rate": target, "rate": best_rate})
        chosen[key] = best
    chosen["_history"] = history
    chosen["_grids"] = {k: list(v) for k, v in grids.items()}
    return chosen


def _positive_rate(packed, thresholds, key) -> float:
    """Fraction of all nearby pairs for which the threshold-driven relation holds."""
    relation = _KEY_RELATION.get(key)
    if relation is None:
        return 0.0
    positives = total = 0
    for rows in packed:
        if not rows:
            continue
        cols = list(zip(*rows))
        values = np.array(cols[1] if relation in ("north_of", "south_of")
                          else cols[0] if relation in ("east_of", "west_of")
                          else cols[2] if relation in ("near", "far")
                          else cols[11])
        margins = _single_relation_margins(relation, cols, thresholds)
        positives += int((margins > 0).sum())
        total += margins.size
    return positives / max(total, 1)


# Which relation each threshold key is read through when measuring selectivity.
_KEY_RELATION = {
    "cardinal_margin": "north_of",
    "near_m": "near",
    "far_m": "far",
    "between_perp_m": None,          # needs a second anchor; set from below
    "between_t_margin": None,
    "along_cos": "along_road",
    "intersection_m": "near_intersection",
    "align_cos": "parallel_to",
    "perp_cos": "perpendicular_to",
    "side_min_m": "same_side_of_road",
}


def _single_relation_margins(relation, cols, thresholds):
    dx = np.array(cols[0]); dy = np.array(cols[1]); fp = np.array(cols[2])
    side_a = np.array(cols[4]); side_b = np.array(cols[5])
    cross = np.array(cols[6], dtype=bool); road_d = np.array(cols[7])
    along = np.array(cols[8]); inter_d = np.array(cols[9])
    align = np.array(cols[10]); parallel = np.array(cols[11])
    if relation == "north_of":
        return dy - thresholds["cardinal_margin"]
    if relation == "near":
        return thresholds["near_m"] - fp
    if relation == "far":
        return fp - thresholds["far_m"]
    if relation == "along_road":
        return np.where(along >= thresholds["along_cos"], along, -np.inf)
    if relation == "near_intersection":
        return thresholds["intersection_m"] - inter_d
    if relation == "parallel_to":
        return parallel - thresholds["align_cos"]
    if relation == "perpendicular_to":
        return thresholds["perp_cos"] - parallel
    if relation == "same_side_of_road":
        floor = thresholds["side_min_m"]
        ok = (np.abs(side_a) >= floor) & (np.abs(side_b) >= floor)
        return np.where(ok, np.sign(side_a) * np.sign(side_b), -np.inf)
    return np.full_like(fp, -np.inf)


def usable_relations(examples) -> dict:
    counts = {}
    for ex in examples:
        counts[ex["relation"]] = counts.get(ex["relation"], 0) + 1
    return counts
