#!/usr/bin/env python3
"""Case galleries for Gate B: where the viewpoint changed the answer.

Reads ``gate_b.json`` and the crops saved during the probe, and writes
``cases_gained`` (top-down wrong, best perspective right) and ``cases_lost``
(top-down right, best perspective wrong).  Each row is the target's crop in every
view, the hardest negative, the text, and the two similarity scores, so a reader
can judge the claim rather than take the aggregate on trust.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

VIEWS = ("topdown", "fpv", "oblique30", "oblique45")
TILE = 224
CAPTION = 44


def rank_of(entry, view, variant, candidate_id):
    data = entry["views"].get(view, {}).get("sims", {}).get(variant)
    if not data:
        return None
    ids = entry["views"][view]["ids"]
    if candidate_id not in ids:
        return None
    sims = np.asarray(data)
    order = np.argsort(-sims)
    rank = int(np.where(order == ids.index(candidate_id))[0][0]) + 1
    best_other = max(s for i, s in enumerate(sims) if ids[i] != candidate_id)
    return {"rank": rank, "target_sim": float(sims[ids.index(candidate_id)]),
            "best_other_sim": float(best_other),
            "margin": float(sims[ids.index(candidate_id)] - best_other),
            "hardest_negative": int(ids[int(np.argmax(
                [s if i != ids.index(candidate_id) else -np.inf
                 for i, s in enumerate(sims)]))])}


def tile(root: Path, entry, view, candidate_id):
    d = root / (f"{entry['split']}_{entry['episode_index']:05d}"
                f"_step{entry['step']:03d}")
    path = d / f"{view}_cand{candidate_id}.png"
    img = cv2.imread(str(path)) if path.exists() else None
    if img is None:
        img = np.full((TILE, TILE, 3), 40, np.uint8)
    return cv2.resize(img, (TILE, TILE))


def build_gallery(cases, root: Path, out_path: Path, variant: str) -> None:
    if not cases:
        return
    rows = []
    for entry, view in cases:
        target = entry["target_id"]
        r = rank_of(entry, view, variant, target)
        neg = r["hardest_negative"] if r else target
        tiles = [tile(root, entry, v, target) for v in VIEWS]
        tiles.append(tile(root, entry, view, neg))
        strip = np.concatenate(tiles, axis=1)

        bar = np.zeros((CAPTION, strip.shape[1], 3), np.uint8)
        text = (f"{entry['split']}:{entry['episode_index']} {entry['map']} | "
                f"'{entry['texts'][variant]}' | d={entry['target_distance']:.0f}m "
                f"| td_rank={rank_of(entry,'topdown',variant,target)['rank'] if rank_of(entry,'topdown',variant,target) else '-'} "
                f"{view}_rank={r['rank'] if r else '-'} "
                f"| margin {r['margin']:+.3f}" if r else "no data")
        cv2.putText(bar, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"cols: topdown | fpv | oblique30 | oblique45 | hardest "
                         f"negative ({entry['map']})",
                    (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1,
                    cv2.LINE_AA)
        rows.append(np.concatenate([bar, strip], axis=0))
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-b", default=None)
    ap.add_argument("--variant", default="phrase", choices=["name", "phrase"])
    ap.add_argument("--per-list", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.gate_b) if args.gate_b else \
        REPO_ROOT / "artifacts" / "gate_b"
    report = json.loads((root / "gate_b.json").read_text())
    rows = report["rows"]

    summary = report["summary"]["main_table"].get(args.variant, {})
    perspective = [v for v in ("fpv", "oblique30", "oblique45") if v in summary]
    if not perspective:
        print("no perspective results for that text variant")
        return
    best_view = max(perspective, key=lambda v: summary[v]["top1"])
    print(f"best perspective for '{args.variant}': {best_view}")

    gained, lost = [], []
    for entry in rows:
        target = entry["target_id"]
        td = rank_of(entry, "topdown", args.variant, target)
        best = rank_of(entry, best_view, args.variant, target)
        if td is None or best is None:
            continue
        if td["rank"] > 1 and best["rank"] == 1:
            gained.append((entry, best_view))
        elif td["rank"] == 1 and best["rank"] > 1:
            lost.append((entry, best_view))

    gained.sort(key=lambda c: -rank_of(c[0], c[1], args.variant,
                                       c[0]["target_id"])["margin"])
    lost.sort(key=lambda c: rank_of(c[0], c[1], args.variant, c[0]["target_id"])["margin"])

    build_gallery(gained[:args.per_list], root / "crops",
                  root / f"cases_gained_{args.variant}.png", args.variant)
    build_gallery(lost[:args.per_list], root / "crops",
                  root / f"cases_lost_{args.variant}.png", args.variant)

    (root / f"cases_{args.variant}.json").write_text(json.dumps({
        "variant": args.variant, "best_perspective": best_view,
        "gained": [{"map": e["map"], "episode_index": e["episode_index"],
                    "step": e["step"], "text": e["texts"][args.variant],
                    "distance": e["target_distance"],
                    "topdown": rank_of(e, "topdown", args.variant, e["target_id"]),
                    "perspective": rank_of(e, best_view, args.variant, e["target_id"])}
                   for e, _ in gained],
        "lost": [{"map": e["map"], "episode_index": e["episode_index"],
                  "step": e["step"], "text": e["texts"][args.variant],
                  "distance": e["target_distance"],
                  "topdown": rank_of(e, "topdown", args.variant, e["target_id"]),
                  "perspective": rank_of(e, best_view, args.variant, e["target_id"])}
                 for e, _ in lost],
    }, indent=2, sort_keys=True, default=float) + "\n")

    print(f"gained (top-down wrong -> {best_view} right): {len(gained)}")
    print(f"lost   (top-down right -> {best_view} wrong): {len(lost)}")
    print(f"wrote {root / f'cases_gained_{args.variant}.png'}")
    print(f"      {root / f'cases_lost_{args.variant}.png'}")


if __name__ == "__main__":
    main()
