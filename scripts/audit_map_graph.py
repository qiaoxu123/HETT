#!/usr/bin/env python3
"""Phase 1: build the hierarchical graph over every block and audit its shape.

Nothing is trained here and no instruction is read.  The question is whether the
map can be turned into the entities the language actually denotes -- in
particular whether a road name is a thing with geometry, rather than a label
sprinkled over a pile of unrelated boxes.

The audit is deliberately suspicious of its own premise.  Grouping road segments
by name only helps if a name is *one* road; if names were reused across a block
the grouping would be wrong, so the number of names that split into several
spatially separate components is counted and reported rather than assumed small.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.anchor_parser import normalise  # noqa: E402
from sensaturban_fpv.config import artifact_dir, load_config, load_landmarks  # noqa: E402
from sensaturban_fpv.spatial_graph import build_block_graph  # noqa: E402
from sensaturban_fpv.spatial_graph.edge_features import (  # noqa: E402
    EDGE_DIM, EDGE_FEATURES, EDGE_INDEX,
)
from sensaturban_fpv.spatial_graph.node_types import NodeKind  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--blocks", type=int, default=0,
                    help="limit the number of blocks (0 = all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "spatial_graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_by_map = load_landmarks(cfg)
    names = sorted(objects_by_map)
    if args.blocks:
        names = names[:args.blocks]

    totals = Counter()
    region_sizes = []
    split_names = Counter()
    per_block = {}
    edge_samples = []
    t0 = time.time()

    for map_name in names:
        graph = build_block_graph(objects_by_map[map_name], map_name)
        summary = graph.stats.summary()
        per_block[map_name] = summary
        for kind, count in summary["by_kind"].items():
            totals[kind] += count
        region_sizes.extend(graph.stats.region_sizes)
        for name, n in graph.stats.region_components.items():
            if name:
                split_names[name] += 1
        totals["blocks"] += 1
        totals["nodes"] += summary["n_nodes"]

        # A few edge vectors, to prove the schema produces finite numbers on
        # real geometry rather than only on the synthetic cases in the tests.
        if len(edge_samples) < 200 and len(graph.nodes) > 2:
            rng = np.random.default_rng(0)
            for _ in range(5):
                a, b = rng.choice(len(graph.nodes), size=2, replace=False)
                edge_samples.append(graph.edge(graph.nodes[a], graph.nodes[b]))
        print(f"  {map_name:24s} nodes {summary['n_nodes']:5d} "
              f"regions {summary['n_road_regions']:3d} "
              f"({time.time() - t0:.0f}s)", flush=True)

    sizes = np.asarray(region_sizes or [0])
    edges = np.stack(edge_samples) if edge_samples else np.zeros((0, EDGE_DIM))
    report = {
        "blocks": totals["blocks"],
        "nodes": totals["nodes"],
        "by_kind": {k: v for k, v in totals.items()
                    if k not in ("blocks", "nodes")},
        "road_regions": {
            "count": int(totals.get(NodeKind.ROAD_REGION.value, 0)),
            "mean_segments": float(sizes.mean()) if sizes.size else 0.0,
            "median_segments": float(np.median(sizes)) if sizes.size else 0.0,
            "max_segments": int(sizes.max()) if sizes.size else 0,
            "p90_segments": float(np.percentile(sizes, 90)) if sizes.size else 0.0,
            "histogram": {str(int(k)): int(v) for k, v in
                          zip(*np.unique(sizes, return_counts=True))},
        },
        "names_that_split_into_components": dict(split_names.most_common(20)),
        "n_names_that_split": len(split_names),
        "edge_schema": {
            "dim": EDGE_DIM,
            "columns": [name for name, _ in EDGE_FEATURES],
            "finite_on_real_edges": bool(np.isfinite(edges).all()),
            "n_sampled": int(edges.shape[0]),
            "abs_max_per_column": ({name: float(np.abs(edges[:, i]).max())
                                    for name, i in EDGE_INDEX.items()}
                                   if edges.shape[0] else {}),
        },
        "per_block": per_block,
    }
    (out_dir / "map_graph_audit.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")

    print(f"\nblocks {report['blocks']}  nodes {report['nodes']}")
    print(f"by kind: {json.dumps(report['by_kind'])}")
    print(f"road regions: {json.dumps(report['road_regions'], default=float)}")
    print(f"names split into >1 component: {len(split_names)}")
    print(f"edge schema dim {EDGE_DIM}, finite on real edges: "
          f"{report['edge_schema']['finite_on_real_edges']}")
    print(f"-> {out_dir / 'map_graph_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
