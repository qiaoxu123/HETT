#!/usr/bin/env python3
"""Qualitative cases: what did each view see, and what did the fusion combine?

A table cannot show whether a joint representation used both views or merely
picked one.  Each figure shows one sample: the referenced entity and the
runner-up it was confused with, each with its tight and context crops in both
views and its masks drawn on them, next to the per-candidate scores each single
view gave them and the rank the fusion gave the target.

The cases are selected by rule from the analysis output -- entities the fusion
got right where no single view did, and failures -- so a reader can re-derive
the same set.  The per-candidate scores come straight out of the stored sample,
so nothing here depends on re-running a model.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from sensaturban_fpv import citynav  # noqa: E402
from sensaturban_fpv.config import (  # noqa: E402
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.entity_geometry import (  # noqa: E402
    EntitySpec, build_entity_payload, draw_mask_overlay,
)
from sensaturban_fpv.pointcloud_renderer import Camera, render_cloud_region  # noqa: E402
from run_gate_b import topdown_raster  # noqa: E402

OBLIQUE_PITCH = -45.0
OBLIQUE_SIZE = 2048
PATCH_GRID = 32
PANEL = 250
COLOURS = {"td": (255, 70, 70), "oblique": (70, 210, 255)}
SINGLE_VIEWS = ("TD_masked", "O_masked", "TD_tight", "O_tight")


def label(canvas, text, origin, colour=(235, 235, 235), scale=0.46):
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                1, cv2.LINE_AA)


def pick_cases(rows, methods, single, fusion, per_bucket):
    """Rank samples into buckets by whether each method found the target."""
    by_key = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["method"] in methods and row["split"] == "val_unseen":
            by_key[row["key"]][row["method"]].append(row)
    buckets = defaultdict(list)
    for key, entries in by_key.items():
        if single not in entries or fusion not in entries:
            continue
        meta = entries[fusion][0]
        # Majority over seeds, matching the transition counts in the analysis --
        # "correct in one of three runs" is not what the table reports.
        single_ok = np.mean([bool(r["top1"]) for r in entries[single]]) >= 0.5
        fusion_ok = np.mean([bool(r["top1"]) for r in entries[fusion]]) >= 0.5
        if fusion_ok and not single_ok:
            buckets[f"{meta['group']}_fusion_only"].append((key, meta))
        elif fusion_ok:
            buckets[f"{meta['group']}_fusion_correct"].append((key, meta))
        else:
            buckets["fusion_failure"].append((key, meta))
    chosen = {}
    for name, items in buckets.items():
        items.sort(key=lambda kv: kv[1]["target_distance"])
        if name.endswith("fusion_correct"):
            # only worth showing when there is no stronger case for the same group
            group = name.split("_")[0]
            if buckets.get(f"{group}_fusion_only"):
                continue
        chosen[name] = items[:per_bucket]
    return chosen


def view_scores(archive, variant):
    """Per-candidate cosine for every frozen single view, straight from the sample."""
    text = archive[f"text_{variant}"]
    out = {}
    for name in SINGLE_VIEWS:
        feature = {"TD_masked": "td_context_masked", "O_masked": "oblique_context_masked",
                   "TD_tight": "td_tight_masked", "O_tight": "oblique_tight_masked"}[name]
        out[name] = archive[feature] @ text
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variant", default="phrase")
    ap.add_argument("--per-bucket", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    results_dir = Path(args.results) if args.results else \
        artifact_dir(cfg) / "entity_grounding" / "results"
    analysis = json.loads((results_dir / f"analysis_{args.variant}.json").read_text())
    out_dir = Path(args.out) if args.out else \
        artifact_dir(cfg) / "entity_grounding" / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in
            (results_dir / f"persample_{args.variant}.jsonl").read_text().splitlines()
            if line.strip()]
    methods = set(analysis["frozen"]) | set(analysis["learned"])
    single = analysis["reference_single"]
    fusion = analysis["reference_fusion"]
    chosen = pick_cases(rows, methods, single, fusion, args.per_bucket)
    for name, items in sorted(chosen.items()):
        print(f"  {name}: {len(items)} cases")
    if not chosen:
        raise SystemExit("no cases selected")

    target_rank = defaultdict(dict)
    for row in rows:
        if row["split"] != "val_unseen" or not row["top1"]:
            continue
        target_rank[(row["method"], row["key"])].setdefault("ranks", []).append(row["rank"])
    for row in rows:
        if row["split"] != "val_unseen":
            continue
        target_rank[(row["method"], row["key"])].setdefault("all", []).append(row["rank"])

    objects_by_map = load_landmarks(cfg)
    episodes = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        for episode in citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split):
            episodes[(episode.map_name, episode.index)] = episode

    cache, rasters = {}, {}
    written = defaultdict(int)
    for bucket, items in sorted(chosen.items()):
        for key, meta in items:
            map_name = meta["map"]
            if map_name not in cache:
                cache = {map_name: build_map_context(cfg, map_name,
                                                     objects_by_map=objects_by_map)}
            ctx = cache[map_name]
            if map_name not in rasters:
                rasters[map_name] = topdown_raster(Path(cfg["paths"]["ortho_dir"]),
                                                   map_name)
            raster_info = rasters[map_name]
            sorted_xyz, _ = ctx.grid.sorted_arrays()
            labels = ctx.grid.sorted_labels()
            episode = episodes.get((map_name, meta["episode_index"]))
            if episode is None:
                continue
            step = int(meta["step"])
            position = np.asarray(episode.trajectory[step, :3], dtype=np.float64)
            camera = Camera(position=position, yaw=float(episode.yaw()[step]),
                            pitch=np.deg2rad(OBLIQUE_PITCH), width=OBLIQUE_SIZE,
                            height=OBLIQUE_SIZE, hfov_deg=cfg["render"]["hfov_deg"],
                            near=cfg["render"]["near"], far=cfg["render"]["far"])
            render = render_cloud_region(ctx.cloud, ctx.grid, camera, splat_radius=0,
                                         lod=cfg["render"]["lod"])

            archive = np.load(results_dir.parent / "data" / key, allow_pickle=True)
            ids = [int(i) for i in archive["entity_ids"]]
            target_index = int(np.flatnonzero(archive["is_target"])[0])
            target_id = ids[target_index]
            scores = view_scores(archive, args.variant)
            # The runner-up is the candidate some single view ranked above the
            # target -- the confusion this sample actually turned on.
            best_other = max(scores, key=lambda n: float(
                max(s for j, s in enumerate(scores[n]) if j != target_index)))
            negative_index = int(max(
                (j for j in range(len(ids)) if j != target_index),
                key=lambda j: max(float(scores[n][j]) for n in SINGLE_VIEWS)))
            negative_id = ids[negative_index]
            del best_other

            specs = [EntitySpec(i, str(t), objects_by_map[map_name][i].position,
                                objects_by_map[map_name][i].dimension,
                                objects_by_map[map_name][i].contour) for i, t in
                     zip(ids, archive["entity_types"])]
            payload, _ = build_entity_payload(ctx.grid, sorted_xyz, labels, specs,
                                              raster_info, camera, render, PATCH_GRID)
            by_id = {p["spec"].entity_id: p for p in payload}
            if target_id not in by_id or negative_id not in by_id:
                continue

            rows_out = []
            for role, entity_id, index in (("GT", target_id, target_index),
                                           ("hard negative", negative_id,
                                            negative_index)):
                item = by_id[entity_id]
                panels = [
                    draw_mask_overlay(item["crops"]["td"]["tight"],
                                      item["view"].mask_of("td", "tight", PATCH_GRID),
                                      COLOURS["td"]),
                    draw_mask_overlay(item["crops"]["oblique"]["tight"],
                                      item["view"].mask_of("oblique", "tight",
                                                           PATCH_GRID),
                                      COLOURS["oblique"]),
                    draw_mask_overlay(item["crops"]["td"]["context"],
                                      item["view"].mask_of("td", "context", PATCH_GRID),
                                      COLOURS["td"]),
                    draw_mask_overlay(item["crops"]["oblique"]["context"],
                                      item["view"].mask_of("oblique", "context",
                                                           PATCH_GRID),
                                      COLOURS["oblique"]),
                ]
                rows_out.append((role, entity_id, index, panels))

            height = 2 * (PANEL + 30)
            canvas = np.full((height, 4 * (PANEL + 10) + 470, 3), 20, np.uint8)
            for r, (role, entity_id, index, panels) in enumerate(rows_out):
                y = r * (PANEL + 30)
                for j, panel in enumerate(panels):
                    x = 6 + j * (PANEL + 10)
                    canvas[y:y + PANEL, x:x + PANEL] = cv2.resize(
                        panel, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
                colour = (120, 235, 120) if role == "GT" else (240, 170, 90)
                label(canvas, f"{role}: {archive['entity_types'][index]} #{entity_id}",
                      (12, y + PANEL + 22), colour)
                x = 6 + 4 * (PANEL + 10) + 8
                for j, view in enumerate(SINGLE_VIEWS):
                    score = float(scores[view][index])
                    rank = int(np.sum(scores[view] > score)) + 1
                    label(canvas, f"{view:10s} score {score:+.4f}  rank {rank}",
                          (x, y + 20 + j * 22), scale=0.44)
            x = 6 + 4 * (PANEL + 10) + 8
            lines = [
                f"{bucket}   {map_name}  ep{meta['episode_index']} step{step}",
                f"instruction: {archive.get('text_phrase', '')}",
                f"candidates: {len(ids)}   distance: {meta['target_distance']:.0f} m",
            ]
            for j, line in enumerate(lines):
                label(canvas, line[:78], (x, 2 * PANEL + 34 + j * 22),
                      (235, 235, 235), 0.44)
            fusion_ranks = target_rank[(fusion, key)].get("all", [])
            if fusion_ranks:
                label(canvas, f"{fusion} target rank "
                              f"{np.mean(fusion_ranks):.1f}", (x, 2 * PANEL + 100),
                      (200, 220, 255), 0.44)

            name = f"{bucket}_{written[bucket]:02d}_{map_name}_ep{meta['episode_index']}"
            cv2.imwrite(str(out_dir / f"{name}.jpg"),
                        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            written[bucket] += 1
            print(f"    wrote {name}", flush=True)

    print(f"\nwrote {sum(written.values())} case figures to {out_dir}")


if __name__ == "__main__":
    main()
