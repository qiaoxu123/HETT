"""Node kinds for the hierarchical map graph, and what a node is allowed to hold.

The previous round ended with a specific structural finding: 41.6% of anchor
phrases name more than one entity, and the worst case is a road -- ``aldridge
road`` is 180 separate road segments. Any design that binds language to *one*
segment has to guess which, and the audit showed there is nothing in the
instruction to guess with. So a road name binds to a **RoadRegion**: the set of
segments sharing that name and forming one connected piece of road. Everything
else stays an instance node.

**What a node may not hold.** ``target flag``, ``candidate index``, ``GT rank``
and ``referenced flag`` are absent by construction rather than by discipline.
The previous rounds each lost time to an arm that had quietly read the answer,
and the cheapest defence is for the node object to have nowhere to put it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class NodeKind(str, Enum):
    """Exactly one kind per node; the partition is total and disjoint."""

    BUILDING = "building"          # object_type Building
    LANDMARK = "landmark"          # named, and not a road or a building
    OBJECT = "object"              # unnamed, not a road or a building
    ROAD_SEGMENT = "road_segment"  # one annotated TrafficRoad piece
    ROAD_REGION = "road_region"    # a name + a connected component of segments
    INTERSECTION = "intersection"  # where two road regions meet

    @property
    def is_instance(self) -> bool:
        return self in (NodeKind.BUILDING, NodeKind.LANDMARK, NodeKind.OBJECT)


# Objects whose SensatUrban class is a road, as CityRefer annotates it.  ``Rail``
# and ``Footpath`` are kept out: the corpus's road language is about drivable
# road, and folding a footpath into "Aldridge Road" would make the region a
# worse approximation of the thing the sentence means.
ROAD_CLASSES = frozenset({"TrafficRoad"})
PATH_CLASSES = frozenset({"Footpath", "Rail", "Bridge"})

# Entity classes that carry a name worth binding language to.
NAMED_CLASSES = frozenset({
    "Building", "TrafficRoad", "Parking", "HighVegetation", "Water",
    "StreetFurniture", "Wall", "Ground", "Bridge", "Rail", "Footpath", "Bike",
})


@dataclass
class Node:
    """One node of the graph.  Geometry only; no label of any kind."""

    node_id: int
    kind: NodeKind
    entity_type: str
    name: str
    center: np.ndarray            # (3,)
    dimension: np.ndarray         # (3,) full extents
    footprint: np.ndarray         # (P, 2) closed polygon in world XY
    major_axis: np.ndarray        # (2,) unit, the footprint's long direction
    member_ids: tuple = ()        # for a region: the segment ids it contains
    parent_region: int = -1       # for a segment: the region it belongs to
    height: float = 0.0

    @property
    def is_region(self) -> bool:
        return self.kind is NodeKind.ROAD_REGION

    @property
    def norm_name(self) -> str:
        from ..anchor_parser import normalise
        return normalise(self.name)

    @property
    def footprint_area(self) -> float:
        if self.footprint.shape[0] < 3:
            return 0.0
        x, y = self.footprint[:, 0], self.footprint[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def bbox(self) -> tuple:
        """Axis-aligned (lo, hi) of the footprint, or of the centre if none."""
        if self.footprint.shape[0] == 0:
            half = np.abs(self.dimension[:2]) / 2.0
            return self.center[:2] - half, self.center[:2] + half
        return self.footprint.min(axis=0), self.footprint.max(axis=0)


@dataclass
class GraphStats:
    """Counts for the audit, filled in as the graph is built."""

    n_nodes: int = 0
    n_edges: int = 0
    by_kind: dict = field(default_factory=dict)
    region_sizes: list = field(default_factory=list)
    unnamed_segments: int = 0
    region_components: dict = field(default_factory=dict)

    def summary(self) -> dict:
        sizes = np.asarray(self.region_sizes or [0])
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "by_kind": dict(self.by_kind),
            "n_road_regions": int(self.by_kind.get(NodeKind.ROAD_REGION.value, 0)),
            "mean_segments_per_region": float(sizes.mean()) if sizes.size else 0.0,
            "max_segments_per_region": int(sizes.max()) if sizes.size else 0,
            "median_segments_per_region": float(np.median(sizes)) if sizes.size else 0.0,
            "regions_from_a_multi_component_name": self.region_components,
            "unnamed_segments_left_as_segments": self.unnamed_segments,
        }
