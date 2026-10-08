#!/usr/bin/env python3
"""Evaluate visual reranking on a saved B0 Top-K JSONL export (no controller)."""
import argparse
import json
from pathlib import Path

import numpy as np

from multiagent.visual_goal.candidate_rerank import candidate_recall, oracle_order, rerank_topk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="JSONL with belief_scores, visual_scores, candidate_xy, target_xy, split")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()
    rows = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    if any(r.get("split") == "test_unseen" for r in rows):
        raise ValueError("test_unseen is prohibited in this experiment")
    result = {"n": len(rows), "alpha": args.alpha, "by_split": {}}
    for split in ("train_seen", "val_seen", "val_unseen"):
        ss = [r for r in rows if r.get("split") == split]
        if not ss:
            continue
        base, fused, oracle = [], [], []
        for row in ss:
            coords = np.asarray(row["candidate_xy"], dtype=float)
            target = np.asarray(row["target_xy"], dtype=float)
            base_order = np.argsort(-np.asarray(row["belief_scores"], dtype=float), kind="stable")
            fused_order = rerank_topk(row["belief_scores"], row["visual_scores"], coords, target, args.alpha)["order"]
            oracle_ix = oracle_order(coords, target)
            base.append(candidate_recall(coords[base_order], target))
            fused.append(candidate_recall(coords[fused_order], target))
            oracle.append(candidate_recall(coords[oracle_ix], target))
        def summarize(vals):
            return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        result["by_split"][split] = {"belief": summarize(base), "visual_rerank": summarize(fused), "oracle_geometry": summarize(oracle)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

