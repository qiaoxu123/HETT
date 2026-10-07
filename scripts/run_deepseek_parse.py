#!/usr/bin/env python3
"""Phase 4/5: map instructions onto the ontology, then onto map entities.

The model sees the sentence and the relation ontology.  It does not see the map,
the target, or any candidate -- :func:`deepseek_parser.check_no_leakage` asserts
that on every request, and the request is built in one place so the assertion
covers everything.

Then the *binding* happens here, not in the model: an anchor's ``entity_name``
is matched against the block's nodes, **road regions first**, because a name
like "Aldridge Road" denotes a road rather than one of its segments.  A name that
matches several instances keeps all of them as hypotheses -- the round never
asks which one is meant, and the reasoner marginalises over the set.

Two runs per instruction, with the ontology list presented in a different order,
give the consistency measurement section 19 asks for: a model reading the
sentence returns the same relations, one reading position does not.

Every call is cached on the request body, so re-running costs nothing, and token
usage is recorded so the round's cost is a number rather than an estimate.
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
from sensaturban_fpv.spatial_graph.deepseek_parser import (  # noqa: E402
    Cache, ONTOLOGY, build_request, call_model, check_no_leakage,
    normalisation_consistency, require_key, unknown_rate, validate,
)
from sensaturban_fpv.spatial_graph.node_types import NodeKind  # noqa: E402


def load_rows(path: Path, split: str, limit: int = 0) -> list:
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    rows = [r for r in rows if r["split"] == split]
    return rows[:limit] if limit else rows


def bind_anchor(entity_name: str, graph) -> dict:
    """Which map nodes a name could denote, road regions first.

    A road name binds to the *region*, which is the round's premise: the
    sentence does not choose among a road's segments and neither does this.  A
    name that matches several region-or-instance nodes returns all of them, with
    the reason recorded, rather than an arbitrary winner.
    """
    norm = normalise(entity_name or "")
    out = {"query": entity_name, "regions": [], "instances": [],
           "matched": False, "ambiguous": False}
    if not norm:
        return out
    for node in graph.nodes:
        if node.kind is NodeKind.ROAD_REGION and node.norm_name == norm:
            out["regions"].append(int(node.node_id))
        elif node.kind.is_instance and node.norm_name == norm:
            out["instances"].append(int(node.node_id))
    # A phrase the annotation spells longer or shorter than the map name still
    # has to reach it; fall back to the node sharing the most content words.
    if not out["regions"] and not out["instances"]:
        words = {w for w in norm.split() if len(w) > 2}
        best, score = [], 0.0
        for node in graph.nodes:
            if node.kind is NodeKind.ROAD_SEGMENT or not node.norm_name:
                continue
            shared = words & {w for w in node.norm_name.split() if len(w) > 2}
            if shared and len(shared) / max(len(words), 1) > score:
                score = len(shared) / max(len(words), 1)
                best = [int(node.node_id)]
            elif shared and len(shared) / max(len(words), 1) == score:
                best.append(int(node.node_id))
        if best:
            for node_id in best:
                node = graph.by_id[node_id]
                (out["regions"] if node.kind is NodeKind.ROAD_REGION
                 else out["instances"]).append(node_id)
            out["matched_by"] = "token_overlap"
    out["matched"] = bool(out["regions"] or out["instances"])
    out["ambiguous"] = len(out["regions"]) + len(out["instances"]) > 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--consistency-limit", type=int, default=60,
                    help="how many instructions get the second, shuffled run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "spatial_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The instruction set is the relation round's, verbatim: same instruction,
    # same target, same candidate order, same split.  Parsing a different sample
    # would make the two rounds' numbers incomparable.
    data = Path(args.data) if args.data else \
        artifact_dir(cfg) / "relation_v2" / "relation_samples.jsonl"

    key = require_key()
    cache = Cache(out_dir / "deepseek_parse_cache")
    rows = load_rows(data, args.split, args.limit)
    print(f"{len(rows)} instructions from {args.split}", flush=True)

    objects_by_map = load_landmarks(cfg)
    graphs = {}
    parsed_rows, consistency = [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        request = build_request(row["instruction"])
        check_no_leakage(request)
        parsed, _ = call_model(request, key, cache)
        ok, reason = validate(parsed)
        record = {"key": row["key"], "map": row["map"], "split": row["split"],
                  "instruction": row["instruction"], "valid": ok,
                  "reason": reason, "parsed": parsed if ok else None}
        if ok and row["map"] not in graphs:
            graphs[row["map"]] = build_block_graph(objects_by_map[row["map"]],
                                                   row["map"])
        if ok:
            graph = graphs[row["map"]]
            record["anchors"] = [
                dict(bind_anchor(a.get("entity_name", ""), graph), raw=a)
                for a in parsed.get("anchors", [])]
        parsed_rows.append(record)

        if i < args.consistency_limit:
            shuffled = build_request(row["instruction"], shuffle_seed=7)
            second, _ = call_model(shuffled, key, cache)
            if validate(second)[0] and ok:
                consistency.append(normalisation_consistency(parsed, second))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s) "
                  f"cache {cache.stats()['hit_rate']:.2f}", flush=True)

    stats = {
        "split": args.split,
        "instructions": len(parsed_rows),
        "valid_parses": int(sum(r["valid"] for r in parsed_rows)),
        "parse_success_rate": (sum(r["valid"] for r in parsed_rows)
                               / max(len(parsed_rows), 1)),
        "unknown_rate": unknown_rate([r["parsed"] for r in parsed_rows
                                      if r["valid"]]),
        "bindings": {
            "anchors": 0, "matched": 0, "ambiguous": 0, "to_a_region": 0,
        },
        "consistency": {
            "n": len(consistency),
            "relations_equal": (float(np.mean([c["relations_equal"]
                                               for c in consistency]))
                                if consistency else None),
            "relation_jaccard": (float(np.mean([c["relation_jaccard"]
                                                for c in consistency]))
                                 if consistency else None),
            "anchor_jaccard": (float(np.mean([c["anchor_jaccard"]
                                              for c in consistency]))
                               if consistency else None),
        },
        "api": cache.stats(),
    }
    for record in parsed_rows:
        for anchor in record.get("anchors", []):
            stats["bindings"]["anchors"] += 1
            stats["bindings"]["matched"] += int(anchor["matched"])
            stats["bindings"]["ambiguous"] += int(anchor["ambiguous"])
            stats["bindings"]["to_a_region"] += int(bool(anchor["regions"]))

    with (out_dir / f"deepseek_parse_{args.split}.jsonl").open("w") as handle:
        for record in parsed_rows:
            handle.write(json.dumps(record) + "\n")
    (out_dir / f"deepseek_parse_{args.split}_stats.json").write_text(
        json.dumps(stats, indent=1, default=float) + "\n")

    print(json.dumps(stats, indent=1, default=float), flush=True)
    print(f"-> {out_dir / f'deepseek_parse_{args.split}.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
