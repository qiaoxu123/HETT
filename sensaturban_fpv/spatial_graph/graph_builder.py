"""Assemble one block into a hierarchical graph.

The graph is built from the **whole block**, not from a candidate list.  The
feature round lost a measurement to exactly that mistake -- a relation block
scored 0.685 over the candidate list and 0.201 over the map, because the list is
drawn around the answer, so a statistic over it describes the sampling rather
than the scene.

Nothing here is told which entity is the target.  The builder takes a block's
annotated objects and produces nodes, regions, intersections and edges; a caller
that wants to score candidates does so afterwards, against a graph that was
built without knowing which of them was the answer.
"""

from __future__ import annotations

import numpy as np

from ..relation_v2 import principal_axis
from .edge_features import EdgeContext, road_frame
from .node_types import (PATH_CLASSES, ROAD_CLASSES, GraphStats, Node, NodeKind)
from .road_region import build_regions, find_intersections


def _footprint(contour, position, dimension) -> np.ndarray:
    pts = np.asarray([[float(p[0]), float(p[1])] for p in (contour or [])],
                     dtype=np.float64)
    if pts.shape[0] >= 3:
        return pts
    # Degenerate annotations exist; fall back to the box so that every node has
    # a footprint and no downstream distance silently becomes NaN.
    x, y = float(position[0]), float(position[1])
    dx, dy = abs(float(dimension[0])) / 2.0, abs(float(dimension[1])) / 2.0
    return np.array([[x - dx, y - dy], [x + dx, y - dy],
                     [x + dx, y + dy], [x - dx, y + dy]])


def classify(object_type: str, name: str) -> NodeKind:
    """The disjoint partition of section 4 of the brief."""
    if object_type in ROAD_CLASSES:
        return NodeKind.ROAD_SEGMENT
    if object_type == "Building":
        return NodeKind.BUILDING
    if name and object_type not in PATH_CLASSES:
        return NodeKind.LANDMARK
    return NodeKind.OBJECT


class BlockGraph:
    """One block's nodes, edges and the context an edge needs to be read in."""

    def __init__(self, map_name, nodes, stats, context, regions, segments):
        self.map_name = map_name
        self.nodes = nodes
        self.stats = stats
        self.context = context
        self.regions = regions
        self.segments = segments
        self.by_id = {n.node_id: n for n in nodes}
        self.by_kind = {}
        for node in nodes:
            self.by_kind.setdefault(node.kind, []).append(node.node_id)

    def nodes_of(self, kind: NodeKind) -> list:
        return [self.by_id[i] for i in self.by_kind.get(kind, [])]

    def named(self) -> list:
        """Nodes a sentence could plausibly name, regions included."""
        return [n for n in self.nodes
                if n.norm_name and (n.kind is NodeKind.ROAD_REGION
                                    or n.kind.is_instance)]

    def road_frame_for(self, node) -> object:
        """The frame of the road region nearest ``node``."""
        best = None
        for idx, region in enumerate(self.regions):
            frame = road_frame(node, region,
                               self.context.road_members.get(idx, []))
            if best is None or frame.distance < best[1].distance:
                best = (idx, frame)
        return None if best is None else best[1]

    def edge(self, a, b) -> np.ndarray:
        from .edge_features import edge_vector
        return edge_vector(a, b, self.context)


def build_block_graph(objects, map_name: str = "") -> BlockGraph:
    """Build the graph for one block of annotated CityRefer objects.

    ``objects`` is the mapping the repo's own loader returns, so the geometry has
    one definition in the tree rather than a second one here.
    """
    stats = GraphStats()
    segments, instances = [], []
    for obj in objects.values():
        name = (obj.name or "").strip()
        kind = classify(obj.object_type, name)
        footprint = _footprint(obj.contour, obj.position, obj.dimension)
        node = Node(
            node_id=int(obj.id),
            kind=kind,
            entity_type=obj.object_type,
            name=name,
            center=np.array(tuple(obj.position), dtype=np.float64),
            dimension=np.array(tuple(obj.dimension), dtype=np.float64),
            footprint=footprint,
            major_axis=(np.asarray(principal_axis(obj.contour), dtype=np.float64)
                        if obj.contour else np.array([1.0, 0.0])),
            height=float(obj.dimension[2]),
        )
        (segments if kind is NodeKind.ROAD_SEGMENT else instances).append(node)

    regions = build_regions(segments, stats)
    # Region ids start above every object id so that a region can never collide
    # with the entity it was built from.
    next_id = max([n.node_id for n in segments + instances], default=0) + 1
    for offset, region in enumerate(regions):
        region.node_id = next_id + offset
    # A segment points at its region, so an edge can tell two pieces of the same
    # road from two pieces of different ones.
    parent = {}
    for region in regions:
        for member in region.member_ids:
            parent[member] = region.node_id
    for seg in segments:
        seg.parent_region = parent.get(seg.node_id, -1)

    intersections = []
    for a, b, where, gap in find_intersections(regions, segments):
        node = Node(node_id=next_id + len(regions) + len(intersections),
                    kind=NodeKind.INTERSECTION,
                    entity_type="Intersection",
                    name=f"{a.name} / {b.name}",
                    center=np.array([where[0], where[1], 0.0]),
                    dimension=np.array([gap, gap, 0.0]),
                    footprint=np.zeros((0, 2)),
                    major_axis=np.array([1.0, 0.0]))
        node.member_ids = (a.node_id, b.node_id)
        intersections.append(node)

    nodes = instances + segments + regions + intersections
    members = {i: [seg for seg in segments if seg.node_id in region.member_ids]
               for i, region in enumerate(regions)}
    context = EdgeContext(road_regions=regions, road_members=members)

    stats.n_nodes = len(nodes)
    stats.by_kind = {}
    for node in nodes:
        stats.by_kind[node.kind.value] = stats.by_kind.get(node.kind.value, 0) + 1

    graph = BlockGraph(map_name, nodes, stats, context, regions, segments)
    # The nearest-road frame per node, computed once and reused by every edge
    # that node takes part in.
    nearest = {}
    for node in nodes:
        frame = graph.road_frame_for(node)
        if frame is not None:
            nearest[id(node)] = frame
    context.nearest_road = nearest
    return graph


def count_edges(graph: BlockGraph, kinds=None) -> int:
    """Directed node pairs the graph would score, optionally filtered by kind."""
    if kinds is None:
        return len(graph.nodes) * (len(graph.nodes) - 1)
    left = [n for k in kinds for n in graph.nodes_of(k)]
    return len(left) * (len(graph.nodes) - 1)
