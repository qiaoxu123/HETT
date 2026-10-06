#!/usr/bin/env python3
"""Recompute Gate B on the candidate set the two views actually share.

The per-view table in ``gate_b.json`` ranks each view over *its own* candidates,
and those differ: a top-down crop exists for every landmark in the block, while
a 90-degree oblique only contains the handful that fall inside it.  Ranking over
ten candidates and over five are not the same task, so that table is biased
against whichever view sees more.

This script recomputes every metric on the intersection of the two views'
candidate sets, which is the comparison the brief asks for, and reports the
McNemar-style discordant counts alongside so the size of the effect can be
judged rather than just its sign.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

VIEWS = ("topdown", "fpv", "oblique30", "oblique45")
PERSPECTIVE = ("fpv", "oblique30", "oblique45")
VARIANTS = ("name", "phrase")


def rank_within(sims, target_index):
    order = np.argsort(-sims)
    pos = int(np.where(order == target_index)[0][0]) + 1
    others = [sims[i] for i in range(len(sims)) if i != target_index]
    return {"rank": pos, "top1": pos == 1, "top4": pos <= 4, "mrr": 1.0 / pos,
            "margin": float(sims[target_index] - (max(others) if others else np.nan))}


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided sign test p-value, for the discordant pairs only."""
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-b", default=None)
    args = ap.parse_args()

    root = Path(args.gate_b) if args.gate_b else REPO_ROOT / "artifacts" / "gate_b"
    path = root / "gate_b.json"
    report = json.loads(path.read_text())
    rows = report["rows"]

    paired = {v: {view: [] for view in PERSPECTIVE} for v in VARIANTS}
    for row in rows:
        target = row["target_id"]
        for variant in VARIANTS:
            for view in PERSPECTIVE:
                a = row["views"].get("topdown")
                b = row["views"].get(view)
                if not a or not b:
                    continue
                if variant not in a.get("sims", {}) or variant not in b.get("sims", {}):
                    continue
                common = [c for c in a["ids"] if c in set(b["ids"])]
                if target not in common or len(common) < 2:
                    continue
                idx = common.index(target)
                sa = np.array([a["sims"][variant][a["ids"].index(c)] for c in common])
                sb = np.array([b["sims"][variant][b["ids"].index(c)] for c in common])
                paired[variant][view].append({
                    "topdown": rank_within(sa, idx),
                    "other": rank_within(sb, idx),
                    "candidates": len(common),
                    "same_class": sum(
                        1 for i, c in enumerate(common)
                        if c != target
                        and row["candidate_types"][row["candidate_ids"].index(c)]
                        == row["target_type"]),
                    "distance": row["target_distance"],
                    "split": row["split"],
                })

    def agg(entries, key):
        m = [e[key] for e in entries]
        return {
            "n": len(m),
            "top1": float(np.mean([x["top1"] for x in m])),
            "top4": float(np.mean([x["top4"] for x in m])),
            "mrr": float(np.mean([x["mrr"] for x in m])),
            "margin": float(np.mean([x["margin"] for x in m])),
        }

    table, transitions, buckets = {}, {}, {}
    for variant in VARIANTS:
        table[variant] = {}
        transitions[variant] = {}
        for view in PERSPECTIVE:
            entries = paired[variant][view]
            if not entries:
                continue
            table[variant][view] = {
                "topdown": agg(entries, "topdown"),
                "perspective": agg(entries, "other"),
                "mean_candidates": float(np.mean([e["candidates"] for e in entries])),
                "top1_delta": (agg(entries, "other")["top1"]
                               - agg(entries, "topdown")["top1"]),
                "margin_delta": (agg(entries, "other")["margin"]
                                 - agg(entries, "topdown")["margin"]),
            }
            gain = sum(1 for e in entries if not e["topdown"]["top1"] and e["other"]["top1"])
            loss = sum(1 for e in entries if e["topdown"]["top1"] and not e["other"]["top1"])
            transitions[variant][view] = {
                "n": len(entries),
                "td_fail_view_success": gain,
                "td_success_view_fail": loss,
                "net": gain - loss,
                "mcnemar_p": binom_two_sided(gain, gain + loss),
            }
            by_class, by_dist = defaultdict(list), defaultdict(list)
            for e in entries:
                by_class["0" if e["same_class"] == 0
                         else ("1-3" if e["same_class"] <= 3 else ">=4")].append(e)
                d = e["distance"]
                by_dist["near<50m" if d < 50
                        else ("medium50-100m" if d < 100 else "far>100m")].append(e)
            buckets.setdefault(variant, {})[view] = {
                "same_class": {k: {"n": len(v),
                                   "topdown_top1": agg(v, "topdown")["top1"],
                                   "perspective_top1": agg(v, "other")["top1"]}
                               for k, v in sorted(by_class.items())},
                "distance": {k: {"n": len(v),
                                 "topdown_top1": agg(v, "topdown")["top1"],
                                 "perspective_top1": agg(v, "other")["top1"]}
                             for k, v in sorted(by_dist.items())},
            }

    summary = {"paired_table": table, "paired_transitions": transitions,
               "paired_buckets": buckets}
    report["paired"] = summary
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    (root / "gate_b_paired.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")

    print("=== paired (same candidate set for both views) ===")
    for variant in VARIANTS:
        for view in PERSPECTIVE:
            t = table.get(variant, {}).get(view)
            if not t:
                continue
            tr = transitions[variant][view]
            print(f"  {variant:7s} {view:10s} n={t['topdown']['n']:3d} "
                  f"cands={t['mean_candidates']:.1f}  "
                  f"topdown top1={t['topdown']['top1']:.3f} "
                  f"-> {view} top1={t['perspective']['top1']:.3f} "
                  f"({t['top1_delta']:+.3f})  "
                  f"gain/loss {tr['td_fail_view_success']}/{tr['td_success_view_fail']} "
                  f"p={tr['mcnemar_p']:.3f}  margin_delta={t['margin_delta']:+.4f}")
    print(f"\nwrote {root / 'gate_b_paired.json'}")


if __name__ == "__main__":
    main()
