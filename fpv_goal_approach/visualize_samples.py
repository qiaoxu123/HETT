from __future__ import annotations

import argparse
import html
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpv_goal_approach.dataset import read_jsonl


def category(row):
    if row["negative_type"] in {"near_goal", "wrong_view", "wrong_language"}:
        return row["negative_type"]
    if row["visual_confirmed_arrival"]:
        return "arrival_positive"
    return None


def build(data: Path, split: str, count: int, seed: int):
    rows = read_jsonl(data / "metadata" / f"{split}.jsonl")
    rng = random.Random(seed)
    selected = []
    for name in ("arrival_positive", "near_goal", "wrong_view", "wrong_language"):
        candidates = [row for row in rows if category(row) == name]
        rng.shuffle(candidates)
        selected.extend((name, row) for row in candidates[:count])
    output = data / "visualizations" / f"{split}_samples.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for name, row in selected:
        image = "../" + row["frames"][-1]
        pose = row["poses"][-1]
        cards.append(f"""
        <article><img src="{html.escape(image)}"><div><b>{name}</b><br>
        {html.escape(row['instruction'])}<br>distance={row['distance_to_goal_m']:.1f}m,
        yaw={pose[3]:.3f}, pitch={pose[4]:.3f}, target bearing={row['target_bearing_rad']:.3f}<br>
        match={row['target_match']}, arrival={row['visual_confirmed_arrival']}, source={row['render_source']}</div></article>
        """)
    output.write_text("""<!doctype html><meta charset="utf-8"><style>
    body{font-family:sans-serif} main{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    article{border:1px solid #ccc;padding:8px} img{width:100%;aspect-ratio:1;object-fit:cover}
    </style><main>""" + "\n".join(cards) + "</main>")
    print(output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="train_seen")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.data, args.split, args.count, args.seed)
