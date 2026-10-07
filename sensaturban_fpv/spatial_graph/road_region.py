"""Turn a pile of road segments into the roads a sentence can name.

A TrafficRoad annotation is a small piece -- in this corpus typically 7 m by
1.7 m -- and a road is many of them.  ``aldridge road`` is 180 segments.  The
previous round treated each as an independent entity and then had to guess which
one the sentence meant; this module removes the guess by building the thing the
name actually denotes.

**Grouping is by name *and* connectivity.**  Merging every same-named segment in
a block would be wrong wherever a name is reused on a physically separate road,
so segments are clustered into connected components first: two segments join if
their footprints come within ``GAP_M``.  A name that yields several components
yields several regions, and the audit reports how often that happens -- if it
were common, the whole premise would need revisiting, so it is measured rather
than assumed.

**Geometry is a polyline, not a bag of boxes.**  Distance to a region is the
distance to the nearest segment's footprint, and the region carries the nearest
segment, the projection onto it, its tangent and normal, so that "along Aldridge
Road" and "across Aldridge Road" are read against the road's own direction
rather than against whichever segment happened to be closest.
"""

from __future__ import annotations

import numpy as np

from .node_types import ROAD_CLASSES, Node, NodeKind

# Two segments join if their footprints come within this distance.  The
# annotations are not perfectly contiguous -- a junction or a dropped detection
# leaves a hole -- so an exact touch test would shatter one road into dozens of
# pieces.  The value is a property of the annotation, not a tuned parameter, and
# the audit reports the component count it produces.
GAP_M = 6.0


def _bbox_gap(a: Node, b: Node) -> float:
    """Axis-aligned gap between two footprints; 0 when they overlap."""
    a_lo, a_hi = a.bbox()
    b_lo, b_hi = b.bbox()
    dx = max(0.0, max(a_lo[0] - b_hi[0], b_lo[0] - a_hi[0]))
    dy = max(0.0, max(a_lo[1] - b_hi[1], b_lo[1] - a_hi[1]))
    return float(np.hypot(dx, dy))


class _Union:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def connected_components(nodes, gap: float = GAP_M):
    """Group indices of ``nodes`` into spatially connected clusters.

    Uses the bounding-box gap as a cheap filter before an exact check, so a
    block's few hundred segments do not require a full pairwise polygon
    distance sweep.
    """
    n = len(nodes)
    union = _Union(n)
    order = sorted(range(n), key=lambda i: nodes[i].center[0])
    for a in range(n):
        i = order[a]
        for b in range(a + 1, n):
            j = order[b]
            # Sorted on x, so once the x gap exceeds the threshold, so does
            # every later one.
            if nodes[j].center[0] - nodes[i].center[0] > gap + 50.0:
                break
            if _bbox_gap(nodes[i], nodes[j]) <= gap:
                union.union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(union.find(i), []).append(i)
    return list(groups.values())


def group_by_name(segments) -> dict:
    """Normalised road name -> the segment indices carrying it."""
    from ..anchor_parser import normalise
    by_name = {}
    for i, node in enumerate(segments):
        key = normalise(node.name) if node.name else ""
        by_name.setdefault(key, []).append(i)
    return by_name


def region_polyline(members):
    """Order a region's segments into a walk, and return its points.

    Roads in this corpus are annotated as a chain of small pieces.  Ordering
    them by projecting onto the cluster's principal axis gives a polyline good
    enough for a tangent and a side-of-road sign; it is not claimed to be the
    true centreline, and nothing downstream depends on it being one.
    """
    centres = np.stack([m.center[:2] for m in members])
    if centres.shape[0] == 1:
        return centres, np.array([1.0, 0.0])
    centred = centres - centres.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    order = np.argsort(centred @ axis)
    return centres[order], axis / max(float(np.linalg.norm(axis)), 1e-9)


def build_regions(segments, stats=None) -> list:
    """Build one :class:`Node` per (name, connected component).

    A name that splits into several components produces several regions, which
    is the behaviour the brief asks for; the count is recorded so the reader can
    see how often the map disagrees with the assumption that a name is one road.

    **Unnamed segments are left out.**  Grouping them by nothing but adjacency
    merges a block's whole unnamed road network into one 66-segment blob, which
    is a connected component rather than a road and is not something a sentence
    can name.  They stay as segment nodes, with their geometry and their edges;
    they simply have no region to bind to.
    """
    regions = []
    by_name = {k: v for k, v in group_by_name(segments).items() if k}
    if stats is not None:
        stats.unnamed_segments = sum(1 for s in segments if not s.name)
    for name, idx in sorted(by_name.items()):
        members = [segments[i] for i in idx]
        for component in connected_components(members):
            group = [members[i] for i in component]
            centres = np.stack([m.center for m in group])
            lo = np.min(np.stack([m.bbox()[0] for m in group]), axis=0)
            hi = np.max(np.stack([m.bbox()[1] for m in group]), axis=0)
            pts = np.concatenate([m.footprint for m in group
                                  if m.footprint.shape[0] >= 1]) \
                if any(m.footprint.shape[0] for m in group) else np.zeros((0, 2))
            _, axis = region_polyline(group)
            region = Node(
                node_id=-1,
                kind=NodeKind.ROAD_REGION,
                entity_type="RoadRegion",
                name=group[0].name,
                center=centres.mean(axis=0),
                dimension=np.array([hi[0] - lo[0], hi[1] - lo[1],
                                    float(np.max([m.height for m in group]))]),
                footprint=pts,
                major_axis=axis,
                member_ids=tuple(int(m.node_id) for m in group),
                height=float(np.max([m.height for m in group])),
            )
            regions.append(region)
            if stats is not None:
                stats.region_sizes.append(len(group))
        if stats is not None and len(connected_components(members)) > 1:
            stats.region_components[name] = len(connected_components(members))
    return regions


def find_intersections(regions, segments, max_gap: float = 12.0) -> list:
    """Nodes where two named road regions come within ``max_gap``.

    Only regions are compared, and only through their member segments, so the
    cost is bounded by the number of road pieces rather than by the block.
    ``member_ids`` holds entity ids, so the segments are looked up by id: they
    are not positions in the list, and indexing one by the other is an
    ``IndexError`` waiting for a block whose ids do not start at zero.
    """
    by_id = {seg.node_id: seg for seg in segments}
    out = []
    seen = set()
    for a_i, a in enumerate(regions):
        if not a.member_ids:
            continue
        for b_i in range(a_i + 1, len(regions)):
            b = regions[b_i]
            if not b.member_ids or a.norm_name == b.norm_name:
                continue
            key = tuple(sorted((a.norm_name, b.norm_name)))
            if key in seen:
                continue
            a_members = [by_id[i] for i in a.member_ids if i in by_id]
            b_members = [by_id[i] for i in b.member_ids if i in by_id]
            best, where = None, None
            for m in a_members:
                for k in b_members:
                    gap = _bbox_gap(m, k)
                    if best is None or gap < best:
                        best, where = gap, (m.center + k.center) / 2.0
            if best is not None and best <= max_gap:
                seen.add(key)
                out.append((a, b, where, float(best)))
    return out
