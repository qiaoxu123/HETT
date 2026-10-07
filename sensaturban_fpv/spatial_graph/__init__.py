"""Hierarchical spatial map graph: regions, geometry, and clean supervision.

The four responsibilities the round keeps apart, and the modules that own them:

* **map entities** -- :mod:`graph_builder`, :mod:`road_region`, :mod:`node_types`
* **geometry** -- :mod:`edge_features`, :mod:`relation_geometry`
* **clean supervision** -- :mod:`synthetic_relations`
* **language** -- :mod:`deepseek_parser`, and the binding in :mod:`binding`

Nothing in this package reads a target flag, a candidate index or a GT rank;
:class:`node_types.Node` has nowhere to put one.
"""

from .edge_features import EDGE_DIM, EDGE_FEATURES, EDGE_INDEX, EdgeContext
from .graph_builder import BlockGraph, build_block_graph
from .node_types import GraphStats, Node, NodeKind

__all__ = ["EDGE_DIM", "EDGE_FEATURES", "EDGE_INDEX", "EdgeContext",
           "BlockGraph", "build_block_graph", "GraphStats", "Node", "NodeKind"]
