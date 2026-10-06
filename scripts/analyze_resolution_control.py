#!/usr/bin/env python3
"""Analysis for the resolution-controlled viewpoint test.

Everything is computed on the paired intersection for whichever two sources are
being compared, never on each source's own candidate set: the sources differ in
which candidates are in frame, and comparing two rankings over different
candidate lists would not be a comparison.

Outputs the main matrix, the resolution-matching comparison, attribute and
same-class and distance buckets, margin distributions, paired transitions with
an exact McNemar test, and the case galleries.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from math import comb
from pathlib import Path

import numpy as np

VIEWS = ("td_native", "td_020", "td_030", "td_match_512", "td_match_1536",
         "td_match_2048", "o512", "o1024", "o1536", "o2048")
PERSPECTIVE = ("o512", "o1024", "o1536", "o2048")

# Attribute vocabulary for the phrase split.  Kept as plain word lists so the
# bucketing is inspectable and reproducible; the report prints the lists.
ATTRIBUTE_WORDS = {
    "color": ["red", "white", "black", "gray", "grey", "blue", "green", "brown",
              "yellow", "orange", "silver", "golden", "pink", "purple"],
    "size": ["large", "big", "small", "tall", "short", "long", "wide", "narrow",
             "high", "low", "huge", "little"],
    "appearance": ["striped", "rectangular", "round", "tower", "roof", "facade",
                   "glass", "brick", "stone", "wooden", "metal", "curved",
                   "sloped", "flat", "pointed", "arched", "triangular",
                   "square", "circular", "cylindrical", "conical", "domed",
                   "tapered", "l-shaped", "t-shaped", "u-shaped", "shaped"],
    "instance": ["car", "parking", "field", "court", "cycle", "bike", "boat",
                 "bus", "van", "truck", "chimney", "entrance", "door", "window",
                 "fence", "gate", "lamp", "bench", "bin", "sign"],
}


def rank_within(sims, target_index):
    order = np.argsort(-sims)
    pos = int(np.where(order == target_index)[0][0]) + 1
    others = [sims[i] for i in range(len(sims)) if i != target_index]
    return {"rank": pos, "top1": pos == 1, "top4": pos <= 4, "mrr": 1.0 / pos,
            "margin": float(sims[target_index] - max(others))}


def mcnemar_exact(gain: int, loss: int) -> float:
    n = gain + loss
    if n == 0:
        return 1.0
    k = min(gain, loss)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(centre - half), float(centre + half))


def _tokens(phrase: str) -> set:
    """Split on anything that is not a letter or digit.

    Matching on space-delimited words alone misses "white/grey" and
    "L-shaped", both of which carry exactly the kind of attribute this split is
    about.
    """
    out = set()
    for piece in re.split(r"[^a-z0-9]+", phrase.lower()):
        if not piece:
            continue
        out.add(piece)
        if piece.endswith("s") and len(piece) > 3:
            out.add(piece[:-1])
    return out


def attribute_buckets(phrase: str) -> dict:
    toks = _tokens(phrase)
    hits = {k: sorted(set(ws) & toks) for k, ws in ATTRIBUTE_WORDS.items()}
    hits = {k: v for k, v in hits.items() if v}
    return {"rich": bool(hits), "categories": sorted(hits), "matched": hits}


def pairwise(entry, a, b, variant):
    """Metrics for two sources on the candidate set they share."""
    va, vb = entry["views"].get(a), entry["views"].get(b)
    if not va or not vb:
        return None
    if variant not in va.get("sims", {}) or variant not in vb.get("sims", {}):
        return None
    common = [c for c in va["ids"] if c in set(vb["ids"])]
    if entry["target_id"] not in common or len(common) < 2:
        return None
    idx = common.index(entry["target_id"])
    sa = np.array([va["sims"][variant][va["ids"].index(c)] for c in common])
    sb = np.array([vb["sims"][variant][vb["ids"].index(c)] for c in common])
    return {"a": rank_within(sa, idx), "b": rank_within(sb, idx),
            "n_candidates": len(common)}


def margin_stats(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {}
    return {
        "n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
        "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
        "positive_ratio": float((v > 0).mean()),
        "std": float(v.std()),
    }


def agg(entries, key):
    m = [e[key] for e in entries]
    return {
        "n": len(m),
        "top1": float(np.mean([x["top1"] for x in m])),
        "top4": float(np.mean([x["top4"] for x in m])),
        "mrr": float(np.mean([x["mrr"] for x in m])),
        **{f"margin_{k}": v for k, v in margin_stats([x["margin"] for x in m]).items()
           if k != "n"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--variant", default="phrase")
    args = ap.parse_args()

    root = Path(args.dir) if args.dir else \
        Path(__file__).resolve().parent.parent / "artifacts" / "resolution_control"
    entries = [json.loads(l) for l in (root / "resolution_control.jsonl").read_text().splitlines() if l]
    print(f"{len(entries)} samples, variant={args.variant}")

    metrics = {"variant": args.variant, "samples": len(entries)}

    # ---- main matrix, each source against td_native on the shared candidates
    matrix = {}
    for view in VIEWS:
        rows = [pairwise(e, "td_native", view, args.variant) for e in entries]
        rows = [r for r in rows if r]
        if not rows:
            continue
        matrix[view] = {
            "reference": agg(rows, "a"),
            "view": agg(rows, "b"),
            "n_paired": len(rows),
            "mean_candidates": float(np.mean([r["n_candidates"] for r in rows])),
        }
        gain = sum(1 for r in rows if not r["a"]["top1"] and r["b"]["top1"])
        loss = sum(1 for r in rows if r["a"]["top1"] and not r["b"]["top1"])
        both_s = sum(1 for r in rows if r["a"]["top1"] and r["b"]["top1"])
        both_f = sum(1 for r in rows if not r["a"]["top1"] and not r["b"]["top1"])
        lo, hi = wilson(gain, gain + loss) if gain + loss else (0.0, 1.0)
        matrix[view]["transition"] = {
            "gain": gain, "loss": loss, "both_success": both_s, "both_fail": both_f,
            "delta_top1": matrix[view]["view"]["top1"] - matrix[view]["reference"]["top1"],
            "delta_margin_median": (matrix[view]["view"]["margin_median"]
                                    - matrix[view]["reference"]["margin_median"]),
            "mcnemar_p": mcnemar_exact(gain, loss),
            "gain_rate_ci95": [lo, hi],
        }
    metrics["main_matrix"] = matrix

    # ---- Q2: matched-resolution comparisons
    matched = {}
    for persp, td_match in (("o512", "td_match_512"), ("o1536", "td_match_1536"),
                            ("o2048", "td_match_2048")):
        rows = [pairwise(e, td_match, persp, args.variant) for e in entries]
        rows = [r for r in rows if r]
        if not rows:
            continue
        gain = sum(1 for r in rows if not r["a"]["top1"] and r["b"]["top1"])
        loss = sum(1 for r in rows if r["a"]["top1"] and not r["b"]["top1"])
        matched[f"{persp}_vs_{td_match}"] = {
            "n_paired": len(rows),
            "td_top1": agg(rows, "a")["top1"],
            "perspective_top1": agg(rows, "b")["top1"],
            "delta_top1": (agg(rows, "b")["top1"]
                           - agg(rows, "a")["top1"]),
            "median_td_gsd_m": float(np.median(
                [e["topdown_resolution_for_source"][td_match] for e in entries
                 if td_match in e["topdown_resolution_for_source"]])),
            "gain": gain, "loss": loss, "mcnemar_p": mcnemar_exact(gain, loss),
        }
    metrics["matched_resolution"] = matched

    # ---- Q1: how much does top-down resolution matter on its own
    res_effect = {}
    for view in ("td_020", "td_030"):
        rows = [pairwise(e, "td_native", view, args.variant) for e in entries]
        rows = [r for r in rows if r]
        if rows:
            res_effect[view] = {
                "n_paired": len(rows),
                "native_top1": agg(rows, "a")["top1"],
                "degraded_top1": agg(rows, "b")["top1"],
                "delta_top1": (agg(rows, "b")["top1"]
                               - agg(rows, "a")["top1"]),
            }
    metrics["topdown_resolution_effect"] = res_effect

    # ---- buckets
    def bucket_report(key_fn, label):
        out = {}
        for view in ("td_native",) + PERSPECTIVE:
            groups = {}
            for e in entries:
                pw = pairwise(e, "td_native", view, args.variant)
                if not pw:
                    continue
                groups.setdefault(key_fn(e), []).append(pw)
            out[view] = {
                k: {"n": len(v),
                    "td_top1": agg(v, "a")["top1"],
                    "view_top1": agg(v, "b")["top1"],
                    "delta": (agg(v, "b")["top1"]
                              - agg(v, "a")["top1"]),
                    "td_margin_median": agg(v, "a")["margin_median"],
                    "view_margin_median": agg(v, "b")["margin_median"]}
                for k, v in sorted(groups.items())}
        metrics[label] = out

    bucket_report(lambda e: "0" if e["same_class_candidates"] == 0
                  else ("1-3" if e["same_class_candidates"] <= 3 else ">=4"),
                  "same_class_buckets")
    bucket_report(lambda e: "near<50m" if e["target_distance"] < 50
                  else ("medium50-100m" if e["target_distance"] < 100 else "far>100m"),
                  "distance_buckets")
    for e in entries:
        e["attributes"] = attribute_buckets(e["texts"].get(args.variant) or "")
    bucket_report(lambda e: "attribute_rich" if e["attributes"]["rich"]
                  else "attribute_poor", "attribute_buckets")

    # sub-categories, reported descriptively
    sub = {}
    for cat in ATTRIBUTE_WORDS:
        rows = [pairwise(e, "td_native", "o2048", args.variant) for e in entries
                if cat in e["attributes"]["categories"]]
        rows = [r for r in rows if r]
        if rows:
            sub[cat] = {"n": len(rows),
                        "td_top1": agg(rows, "a")["top1"],
                        "o2048_top1": agg(rows, "b")["top1"]}
    metrics["attribute_subcategories"] = sub

    # ---- margin distribution for each source against itself
    metrics["margin_distribution"] = {
        view: margin_stats([pairwise(e, "td_native", view, args.variant)["b"]["margin"]
                            for e in entries
                            if pairwise(e, "td_native", view, args.variant)])
        for view in VIEWS}

    # ---- exact p-values are recorded per comparison above; also record the
    # per-source positive-margin ratio for the whole candidate set
    metrics["positive_margin_ratio"] = {
        view: float(np.mean([r["b"]["margin"] > 0 for e in entries
                             for r in [pairwise(e, "td_native", view, args.variant)]
                             if r]))
        for view in VIEWS}

    for sub_dir in ("resolution_control", "attribute_analysis", "paired_results",
                    "qualitative_resolution"):
        (root / sub_dir).mkdir(parents=True, exist_ok=True)

    suffix = "" if args.variant == "phrase" else f"_{args.variant}"
    (root / "resolution_control" / f"metrics{suffix}.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=float) + "\n")
    (root / "resolution_control" / "resolution_config.json").write_text(json.dumps({
        "oblique_pitch_deg": -45.0,
        "oblique_sizes_px": [512, 1024, 1536, 2048],
        "hfov_deg": 90.0,
        "topdown_native_resolution_m": 0.1,
        "topdown_fixed_degradations_m": [0.2, 0.3],
        "topdown_matched_to": {"td_match_512": "o512 ground sample distance",
                               "td_match_1536": "o1536 ground sample distance",
                               "td_match_2048": "o2048 ground sample distance"},
        "crop_extent_m": "clip(2.5 * max(dimension), 24, 150), identical for every source",
        "encoder": "google/siglip2-so400m-patch16-512, frozen",
        "paired_subset": "the 158 val_unseen samples paired in Gate B for phrase/oblique45",
    }, indent=2, sort_keys=True) + "\n")

    with (root / "resolution_control" / f"metrics{suffix}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["view", "reference", "n_paired", "td_top1", "view_top1",
                         "delta_top1", "view_top4", "view_mrr", "view_margin_mean",
                         "view_margin_median", "view_margin_positive_ratio",
                         "gain", "loss", "mcnemar_p"])
        for view, m in matrix.items():
            t = m["transition"]
            writer.writerow([view, "td_native", m["n_paired"],
                             round(m["reference"]["top1"], 4), round(m["view"]["top1"], 4),
                             round(t["delta_top1"], 4), round(m["view"]["top4"], 4),
                             round(m["view"]["mrr"], 4), round(m["view"]["margin_mean"], 4),
                             round(m["view"]["margin_median"], 4),
                             round(m["view"]["margin_positive_ratio"], 4),
                             t["gain"], t["loss"], round(t["mcnemar_p"], 4)])

    (root / "paired_results" / f"paired_samples{suffix}.json").write_text(json.dumps(
        [{"split": e["split"], "map": e["map"], "episode_index": e["episode_index"],
          "step": e["step"], "target_id": e["target_id"],
          "candidate_ids": e["candidate_ids"],
          "texts": e["texts"], "distance": e["target_distance"],
          "same_class_candidates": e["same_class_candidates"],
          "attributes": e["attributes"]} for e in entries],
        indent=2, sort_keys=True, default=float) + "\n")
    (root / "attribute_analysis" / f"attribute_split{suffix}.json").write_text(json.dumps(
        {"vocabulary": ATTRIBUTE_WORDS,
         "per_sample": {f"{e['split']}:{e['episode_index']}": e["attributes"]
                        for e in entries}},
        indent=2, sort_keys=True) + "\n")

    print(json.dumps({"main_matrix": {k: {"td": round(v["reference"]["top1"], 4),
                                          "view": round(v["view"]["top1"], 4),
                                          "delta": round(v["transition"]["delta_top1"], 4),
                                          "gain": v["transition"]["gain"],
                                          "loss": v["transition"]["loss"],
                                          "p": round(v["transition"]["mcnemar_p"], 4)}
                                     for k, v in matrix.items()},
                      "matched": matched,
                      "topdown_resolution_effect": res_effect},
                     indent=2, default=float))
    print(f"\nwrote {root / 'resolution_control' / f'metrics{suffix}.json'}")


if __name__ == "__main__":
    main()
