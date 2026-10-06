#!/usr/bin/env python3
"""Qualitative galleries for the resolution-controlled test.

Four lists, ten cases each:

``td_fail_o1536_success``  the viewpoint changed the answer
``o512_fail_o1536_success`` resolution alone changed the answer
``attribute_rich_success`` where a described attribute was the deciding evidence
``o1536_fail``             what still fails, which bounds what resolution buys

Each row shows the target's crop under top-down native, top-down at 0.3 m/px,
oblique45 at 512 / 1024 / 1536, and the hardest negative in both top-down
native and oblique45 1536, so the comparison that produced the number is visible
next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_resolution_control import attribute_buckets  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

TILE = 200
CAPTION = 58
PANELS = ("td_native", "td_030", "o512", "o1024", "o1536")


def rank_in(entry, view, variant, cid):
    data = entry["views"].get(view, {}).get("sims", {}).get(variant)
    if not data:
        return None
    ids = entry["views"][view]["ids"]
    if cid not in ids:
        return None
    sims = np.asarray(data)
    i = ids.index(cid)
    order = np.argsort(-sims)
    rank = int(np.where(order == i)[0][0]) + 1
    others = [(j, s) for j, s in enumerate(sims) if j != i]
    j, best = max(others, key=lambda t: t[1]) if others else (None, np.nan)
    return {"rank": rank, "sim": float(sims[i]), "best_other": float(best),
            "margin": float(sims[i] - best), "hardest_negative": ids[j] if j is not None else None}


def tile(root: Path, entry, view, cid):
    d = root / "crops" / (f"{entry['split']}_{entry['episode_index']:05d}"
                          f"_step{entry['step']:03d}")
    img = cv2.imread(str(d / f"{view}_cand{cid}.png"))
    if img is None:
        img = np.full((TILE, TILE, 3), 40, np.uint8)
    return cv2.resize(img, (TILE, TILE))


def gallery(root: Path, cases, out_path: Path, variant: str) -> None:
    if not cases:
        return
    rows = []
    for entry, note in cases:
        target = entry["target_id"]
        r1536 = rank_in(entry, "o1536", variant, target) or {}
        neg = r1536.get("hardest_negative") or target
        panels = [tile(root, entry, v, target) for v in PANELS]
        panels.append(tile(root, entry, "td_native", neg))
        panels.append(tile(root, entry, "o1536", neg))
        strip = np.concatenate(panels, axis=1)

        bar = np.zeros((CAPTION, strip.shape[1], 3), np.uint8)
        phrase = entry["texts"].get(variant) or ""
        name = entry["texts"].get("name") or entry["target_type"]
        rank_bits = " ".join(
            f"{v}={rank_in(entry, v, variant, target)['rank'] if rank_in(entry, v, variant, target) else '-'}"
            for v in PANELS)
        line1 = (f"{entry['split']}:{entry['episode_index']} {entry['map']} | "
                 f"'{phrase}' | name: {name} | d={entry['target_distance']:.0f}m | "
                 f"same-class={entry['same_class_candidates']}")
        line2 = (f"ranks  {rank_bits}   margin@o1536={r1536.get('margin', float('nan')):+.4f}   "
                 # JSON round-trips dict keys to strings, so the id has to be
                 # looked up as one.
                 f"target_px_ratio={entry['landmark_pixel_ratio_ideal'].get(str(target), float('nan')):.4f}   {note}")
        line3 = ("columns:  td native | td 0.3 m/px | o512 | o1024 | o1536 | "
                 f"hardest negative @td native (id {neg}) | hardest negative @o1536 (id {neg})")
        for i, text in enumerate((line1, line2, line3)):
            cv2.putText(bar, text, (5, 15 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (255, 255, 255) if i == 0 else (170, 170, 170), 1, cv2.LINE_AA)
        rows.append(np.concatenate([bar, strip], axis=0))
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))
    print(f"{out_path.name}: {len(rows)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--variant", default="phrase")
    ap.add_argument("--per-list", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.dir) if args.dir else \
        Path(__file__).resolve().parent.parent / "artifacts" / "resolution_control"
    entries = [json.loads(l) for l in
               (root / "resolution_control.jsonl").read_text().splitlines() if l]
    # The attribute buckets are recomputed here rather than read back: the
    # analysis script attaches them to its own in-memory copy, not to the file.
    for e in entries:
        e["attributes"] = attribute_buckets(e["texts"].get(args.variant) or "")
    out = root / "qualitative_resolution"
    out.mkdir(parents=True, exist_ok=True)

    def ok(e, view):
        return rank_in(e, view, args.variant, e["target_id"])

    td_fail_o_ok, o512_fail_o1536_ok, attr_ok, still_fail = [], [], [], []
    for e in entries:
        td = ok(e, "td_native")
        o512 = ok(e, "o512")
        o1536 = ok(e, "o1536")
        if not (td and o1536):
            continue
        if td["rank"] > 1 and o1536["rank"] == 1:
            td_fail_o_ok.append((e, "viewpoint gain"))
        if o512 and o512["rank"] > 1 and o1536["rank"] == 1:
            o512_fail_o1536_ok.append((e, "resolution gain"))
        if o1536["rank"] == 1 and e.get("attributes", {}).get("rich"):
            attr_ok.append((e, ",".join(e["attributes"]["categories"])))
        if o1536["rank"] > 1:
            still_fail.append((e, f"rank {o1536['rank']}"))

    td_fail_o_ok.sort(key=lambda c: -ok(c[0], "o1536")["margin"])
    o512_fail_o1536_ok.sort(key=lambda c: -(ok(c[0], "o1536")["rank"] == 1))
    attr_ok.sort(key=lambda c: -ok(c[0], "o1536")["margin"])
    still_fail.sort(key=lambda c: ok(c[0], "o1536")["margin"])

    gallery(root, td_fail_o_ok[: args.per_list],
            out / f"td_fail_o1536_success_{args.variant}.png", args.variant)
    gallery(root, o512_fail_o1536_ok[: args.per_list],
            out / f"o512_fail_o1536_success_{args.variant}.png", args.variant)
    gallery(root, attr_ok[: args.per_list],
            out / f"attribute_rich_success_{args.variant}.png", args.variant)
    gallery(root, still_fail[: args.per_list],
            out / f"o1536_still_fails_{args.variant}.png", args.variant)

    print(json.dumps({"td_fail_o1536_success": len(td_fail_o_ok),
                      "o512_fail_o1536_success": len(o512_fail_o1536_ok),
                      "attribute_rich_success": len(attr_ok),
                      "o1536_still_fails": len(still_fail)}, indent=2))


if __name__ == "__main__":
    main()
