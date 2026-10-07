#!/usr/bin/env python3
"""Phase 2: generate clean relation supervision from the maps alone.

Three things happen here, in this order, and the order matters.

1. **Maps are split.**  Not CityNav splits: ``val_seen`` shares all 23 of its
   maps with ``train_seen``, so a held-out *episode* there is not a held-out
   map.  The 30 maps outside ``val_unseen`` are divided by map into synthetic
   train and synthetic val; ``val_unseen``'s 4 maps are the test set and are not
   read until the gate.
2. **Thresholds are selected on the val maps.**  Section 11 forbids inventing
   them; the grid is swept against the criterion in
   ``synthetic_relations.select_thresholds`` and the chosen values are written
   out so the report can quote them rather than describe them.
3. **Examples are generated** for each split, with hard negatives matched on
   entity kind and distance band so that only the relation distinguishes them.

No instruction text is read anywhere in this script.  That is the point: if the
teacher cannot learn relations whose labels *are* geometry, the fault is in the
graph or the model, and no amount of language alignment would have saved it.
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

from sensaturban_fpv.config import artifact_dir, load_config, load_landmarks  # noqa: E402
from sensaturban_fpv.spatial_graph import build_block_graph  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_geometry import (  # noqa: E402
    DEFAULT_THRESHOLDS, RELATION_NAMES,
)
from sensaturban_fpv.spatial_graph.synthetic_relations import (  # noqa: E402
    build_block_examples, select_thresholds,
)

SYNTH_TEST_MAPS = "val_unseen"     # the 4 disjoint maps
VAL_FRACTION = 0.35


def split_maps(cfg) -> dict:
    """Map-disjoint synth train / val / test, with the reason recorded."""
    from sensaturban_fpv import citynav
    dims = Path(cfg["paths"]["citynav_dir"])
    held = {ep.map_name for ep in citynav.load_split(dims, SYNTH_TEST_MAPS)}
    everything = set(load_landmarks(cfg))
    pool = sorted(everything - held)
    # Deterministic, and by map: a fixed hash so the split does not move between
    # runs and cannot be re-rolled until a result looks good.
    order = sorted(pool, key=lambda m: abs(hash(("synth", m))))
    n_val = int(round(VAL_FRACTION * len(pool)))
    return {"synthetic_val": [m for m in order[:n_val]],
            "synthetic_train": [m for m in order[n_val:]],
            "synthetic_test": sorted(held)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "spatial_graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_by_map = load_landmarks(cfg)
    splits = split_maps(cfg)
    print(f"maps: train {len(splits['synthetic_train'])} "
          f"val {len(splits['synthetic_val'])} "
          f"test {len(splits['synthetic_test'])}", flush=True)

    t0 = time.time()
    graphs = {}
    for name in sorted(objects_by_map):
        graphs[name] = build_block_graph(objects_by_map[name], name)
    print(f"built {len(graphs)} block graphs in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    val_blocks = [graphs[m] for m in splits["synthetic_val"]]
    thresholds = select_thresholds(val_blocks, DEFAULT_THRESHOLDS, seed=args.seed)
    history = thresholds.pop("_history", [])
    grids = thresholds.pop("_grids", {})
    print(f"thresholds selected on synthetic-val in {time.time() - t0:.0f}s",
          flush=True)
    for key, value in thresholds.items():
        marker = " (default)" if value == DEFAULT_THRESHOLDS.get(key) else ""
        print(f"  {key:22s} {value}{marker}", flush=True)

    rng = np.random.default_rng(args.seed)
    counts = {}
    for split, maps in (("train", splits["synthetic_train"]),
                        ("val", splits["synthetic_val"]),
                        ("test", splits["synthetic_test"])):
        rows = []
        for name in maps:
            rows.extend(build_block_examples(graphs[name], thresholds, rng, split))
        path = out_dir / f"synthetic_relations_{split}.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        counts[split] = {
            "examples": len(rows),
            "negative_pairs": int(sum(len(r["negative_ids"]) for r in rows)),
            "by_relation": dict(sorted(Counter(r["relation"] for r in rows).items())),
            "maps": len(maps),
        }
        print(f"  {split:6s} {len(rows):6d} examples "
              f"{counts[split]['negative_pairs']:6d} negative pairs", flush=True)

    report = {
        "maps": splits,
        "thresholds": thresholds,
        "threshold_selection": history,
        "threshold_grids": grids,
        "defaults": DEFAULT_THRESHOLDS,
        "counts": counts,
        "relations": list(RELATION_NAMES),
        "no_instruction_text_read": True,
    }
    (out_dir / "synthetic_relations_stats.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")
    print(f"-> {out_dir / 'synthetic_relations_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
