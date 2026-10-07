#!/usr/bin/env python3
"""Parse instructions into a distribution over (relation, frame, anchor).

The previous rounds asked the model for one relation and got one, then measured
what that commitment cost.  This asks for a distribution and takes it seriously:
probabilities are carried through to the geometry and marginalised there, and a
model that is unsure says so with a flat distribution and a low confidence rather
than by guessing.

Two runs per instruction with the relation list presented in a different order,
so the consistency section measures whether the distribution is a property of the
sentence or of the prompt's layout.  Section 17 asks for the distribution
comparison specifically, not just the argmax.

Everything is cached on the request body; the key is read from the environment
and never written anywhere.
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

from sensaturban_fpv.config import artifact_dir, load_config  # noqa: E402
from sensaturban_fpv.spatial_graph.deepseek_parser import (  # noqa: E402
    Cache, build_hypothesis_request, call_model, check_no_leakage, require_key,
    validate_hypotheses,
)
from sensaturban_fpv.spatial_graph.relation_belief import (  # noqa: E402
    clean_hypotheses, hypothesis_confidence, normalised_entropy,
)

SPLITS = ("val_seen", "val_unseen")


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence between two hypothesis distributions.

    Matched by relation *and* frame, so a model that finds the same relation
    under a different frame is counted as disagreeing -- which is the point,
    since the frame is what the geometry will be read against.
    """
    def keyed(hypotheses):
        out = {}
        for h in hypotheses:
            key = (h["relation"], h["frame"])
            out[key] = out.get(key, 0.0) + h["probability"]
        return out

    a, b = keyed(p), keyed(q)
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    pa = np.array([a.get(k, 0.0) for k in keys])
    pb = np.array([b.get(k, 0.0) for k in keys])
    pa, pb = pa / max(pa.sum(), 1e-12), pb / max(pb.sum(), 1e-12)
    m = 0.5 * (pa + pb)

    def kl(x, y):
        mask = (x > 0) & (y > 0)
        return float((x[mask] * np.log(x[mask] / y[mask])).sum())

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--consistency-per-split", type=int, default=400)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "spatial_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = Path(args.data) if args.data else \
        artifact_dir(cfg) / "relation_v2" / "relation_samples.jsonl"

    key = require_key()
    cache = Cache(out_dir / "deepseek_belief_cache")
    rows = [json.loads(line) for line in data.read_text().splitlines()
            if line.strip()]

    stats = {}
    for split in SPLITS:
        subset = [r for r in rows if r["split"] == split]
        if not subset:
            continue
        parsed_rows, consistency = [], []
        t0 = time.time()
        for i, row in enumerate(subset):
            request = build_hypothesis_request(row["instruction"],
                                               top_k=args.top_k)
            check_no_leakage(request)
            answer, _ = call_model(request, key, cache)
            ok, reason = validate_hypotheses(answer) if answer else (False, "null")
            record = {"key": row["key"], "map": row["map"], "split": split,
                      "instruction": row["instruction"], "valid": ok,
                      "reason": reason,
                      "anchors": (answer or {}).get("anchors", []),
                      "hypotheses": clean_hypotheses(
                          (answer or {}).get("hypotheses"), args.top_k) if ok
                      else []}
            record["confidence"] = hypothesis_confidence(record["hypotheses"])
            parsed_rows.append(record)

            if i < args.consistency_per_split:
                second_req = build_hypothesis_request(row["instruction"],
                                                      shuffle_seed=7,
                                                      top_k=args.top_k)
                second, _ = call_model(second_req, key, cache)
                if second and validate_hypotheses(second)[0] and ok:
                    alt = clean_hypotheses(second.get("hypotheses"), args.top_k)
                    top1 = (lambda h: (h[0]["relation"], h[0]["frame"])
                            if h else None)
                    keys = {(h["relation"], h["frame"]) for h in
                            record["hypotheses"]}
                    other = {(h["relation"], h["frame"]) for h in alt}
                    consistency.append({
                        "top1_equal": top1(record["hypotheses"]) == top1(alt),
                        "jaccard": len(keys & other) / max(len(keys | other), 1),
                        "js": js_divergence(record["hypotheses"], alt),
                        "confidence_delta": abs(
                            record["confidence"] - hypothesis_confidence(alt)),
                    })
            if (i + 1) % 100 == 0:
                print(f"  {split} {i + 1}/{len(subset)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        confidences = [r["confidence"] for r in parsed_rows if r["valid"]]
        entropy_by_relation = {}
        for r in parsed_rows:
            for h in r["hypotheses"]:
                entropy_by_relation.setdefault(h["relation"], []).append(
                    normalised_entropy([x["probability"] for x in r["hypotheses"]]))
        stats[split] = {
            "instructions": len(parsed_rows),
            "valid": int(sum(r["valid"] for r in parsed_rows)),
            "success_rate": sum(r["valid"] for r in parsed_rows)
            / max(len(parsed_rows), 1),
            "mean_hypotheses": float(np.mean([len(r["hypotheses"])
                                              for r in parsed_rows if r["valid"]]))
            if any(r["valid"] for r in parsed_rows) else 0.0,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "median_confidence": float(np.median(confidences)) if confidences else 0.0,
            "relation_counts": dict(Counter(h["relation"] for r in parsed_rows
                                            for h in r["hypotheses"])),
            "frame_counts": dict(Counter(h["frame"] for r in parsed_rows
                                         for h in r["hypotheses"])),
            "mean_entropy_by_relation": {
                k: float(np.mean(v)) for k, v in entropy_by_relation.items()},
            "anchor_count": int(sum(len(r["anchors"]) for r in parsed_rows)),
            "consistency": {
                "n": len(consistency),
                "top1_equal": float(np.mean([c["top1_equal"] for c in consistency]))
                if consistency else None,
                "topk_jaccard": float(np.mean([c["jaccard"] for c in consistency]))
                if consistency else None,
                "js_divergence": float(np.mean([c["js"] for c in consistency]))
                if consistency else None,
                "confidence_delta": float(np.mean([c["confidence_delta"]
                                                   for c in consistency]))
                if consistency else None,
            },
            "api": cache.stats(),
        }
        with (out_dir / f"belief_parse_{split}.jsonl").open("w") as handle:
            for record in parsed_rows:
                handle.write(json.dumps(record) + "\n")
        print(f"{split}: {json.dumps(stats[split], default=float)[:400]}", flush=True)

    (out_dir / "belief_parse_stats.json").write_text(
        json.dumps(stats, indent=1, default=float) + "\n")
    print(f"-> {out_dir / 'belief_parse_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
