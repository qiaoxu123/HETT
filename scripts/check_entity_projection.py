#!/usr/bin/env python3
"""Is the 3D anchor right for every kind of entity, not just for buildings?

Buildings are forgiving: they are large, tall, and a bounding box that misses by
a metre still lands on the same roof.  Cars are not.  A car is 4.5 m long in a
block where the surrounding road surface has three hundred points per square
metre, so a support that is really "the box" is really "a patch of road", and a
crop that is really "the street" can look plausible in a contact sheet while
being useless.

So this checks the ladder per type on purpose: it samples candidates grouped by
object type, builds their support, projects it into both views, and reports --
next to the picture -- which rung the support came from, how many points it
holds, what fraction of the entity's own points the oblique view actually sees,
and how many patches each mask covers at each scale.

The figures are meant to be read: the tight crop has to contain the entity, the
masks have to sit on the entity in both views, and a car's mask must not cover
the whole street.  If the correspondence is visibly wrong, the round stops here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    EntitySpec, build_entity_payload, correspondence_coherence,
    draw_mask_overlay, entity_scales, shuffled_correspondence,
)
from sensaturban_fpv.pointcloud_renderer import Camera, render_cloud_region  # noqa: E402
from run_gate_b import topdown_raster  # noqa: E402

OBLIQUE_PITCH = -45.0
OBLIQUE_SIZE = 2048
PATCH_GRID = 32
PANEL = 300
MASK_COLOURS = {"td": (255, 60, 60), "oblique": (60, 220, 255)}
# Types the round has to speak about separately.  Everything else is pooled: the
# point of the grouping is to catch a method that only works on buildings.
GROUPS = {"building": ("Building",),
          "vehicle": ("Car", "Bike"),
          "other": ("StreetFurniture", "Wall", "Parking", "TrafficRoad",
                    "Footpath", "HighVegetation", "Ground", "Water", "Bridge",
                    "Rail")}


def fit(image, side=PANEL):
    return cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)


def label(canvas, text, origin, colour=(255, 255, 255), scale=0.5):
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                1, cv2.LINE_AA)


def make_figure(rows, path, title):
    header, row_h = 44, PANEL + 30
    width = 4 * (PANEL + 10) + 470
    canvas = np.full((header + row_h * len(rows), width, 3), 20, np.uint8)
    label(canvas, title, (10, 28), (240, 240, 240), 0.55)
    for i, row in enumerate(rows):
        y0 = header + i * row_h
        for j, key in enumerate(("td_tight", "oblique_tight", "td_ctx", "oblique_ctx")):
            x = 6 + j * (PANEL + 10)
            canvas[y0:y0 + PANEL, x:x + PANEL] = fit(row["panels"][key])
        tag = "TARGET" if row["is_target"] else "distractor"
        colour = (90, 220, 90) if row["is_target"] else (205, 205, 205)
        label(canvas, f"{row['object_type']} #{row['entity_id']} {tag} "
                      f"d={row['distance_m']:.0f}m", (12, y0 + PANEL + 22), colour)
        x = 6 + 4 * (PANEL + 10) + 8
        lines = [
            f"support : {row['support_source']}  n={row['support_points']}",
            f"extent  : tight {row['tight_extent_m']:.1f} m  "
            f"context {row['context_extent_m']:.1f} m",
            f"TD mask : tight {row['td_tight_patches']:3d}  "
            f"ctx {row['td_ctx_patches']:3d} patches",
            f"O  mask : tight {row['o_tight_patches']:3d}  "
            f"ctx {row['o_ctx_patches']:3d} patches",
            f"O visible points : {row['visible_point_ratio'] * 100:.1f}%",
            f"correspondence   : {row['correspondence_pairs']} pairs",
            f"coherence        : {row['coherence']:.2f} / shuffled "
            f"{row['coherence_shuffled']:.2f}",
        ]
        for j, text in enumerate(lines):
            label(canvas, text, (x, y0 + 22 + j * 26), scale=0.46)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def sample_entities(objects_by_map, group, per_group, max_per_map=6, rng=None):
    """Annotated entities of one type group, spread over maps."""
    wanted = set(GROUPS[group])
    rng = rng or np.random.default_rng(0)
    by_map = defaultdict(list)
    for map_name, objects in objects_by_map.items():
        for obj in objects.values():
            if obj.object_type in wanted:
                by_map[map_name].append(obj)
    chosen = []
    for map_name in sorted(by_map):
        pool = by_map[map_name]
        idx = rng.permutation(len(pool))[:min(max_per_map, len(pool))]
        chosen += [(map_name, pool[int(i)]) for i in idx]
    rng.shuffle(chosen)
    return chosen[:per_group]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--max-per-map", type=int, default=4)
    ap.add_argument("--patch-grid", type=int, default=PATCH_GRID)
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "entity_grounding" / "check"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_by_map = load_landmarks(cfg)
    selected = {}
    for group in args.groups:
        selected[group] = sample_entities(objects_by_map, group, args.per_group,
                                          args.max_per_map)
    split_episodes = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        for episode in citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split):
            split_episodes.setdefault(episode.map_name, []).append(episode)

    records, figure_rows, figure_index = [], [], 0
    t0 = time.time()
    for group, entries in selected.items():
        # Figures are flushed per group, so a batch is never captioned with
        # another group's type.
        figure_rows, figure_index = [], 0
        for map_name, obj in entries:
            episodes = split_episodes.get(map_name, [])
            if not episodes:
                continue
            rng = np.random.default_rng(abs(hash((map_name, obj.id))) % (2 ** 31))
            episode = episodes[int(rng.integers(len(episodes)))]
            position = episode.trajectory[:, :3]
            target = np.asarray(obj.position, dtype=np.float64)
            best = None
            for step in np.linspace(0, len(position) - 1, num=8).astype(int):
                distance = float(np.linalg.norm(target - position[step]))
                if 25.0 <= distance <= 220.0:
                    best = (int(step), distance)
                    break
            if best is None:
                continue
            step, distance = best
            # Aim at the entity rather than using the trajectory's heading: the
            # question here is whether the *projection* is right, and a pose that
            # happens to face away would confound it with observability.  The
            # data pipeline keeps the instruction-following heading, and reports
            # in-frame rate separately.
            delta = target[:2] - position[step][:2]
            yaw_step = float(np.arctan2(delta[1], delta[0]))

            ctx = build_map_context(cfg, map_name, objects_by_map=objects_by_map)
            sorted_xyz, _ = ctx.grid.sorted_arrays()
            labels = ctx.grid.sorted_labels()
            raster_info = topdown_raster(Path(cfg["paths"]["ortho_dir"]), map_name)
            camera = Camera(position=position[step], yaw=yaw_step,
                            pitch=np.deg2rad(OBLIQUE_PITCH), width=OBLIQUE_SIZE,
                            height=OBLIQUE_SIZE, hfov_deg=cfg["render"]["hfov_deg"],
                            near=cfg["render"]["near"], far=cfg["render"]["far"])
            render = render_cloud_region(ctx.cloud, ctx.grid, camera, splat_radius=0,
                                         lod=cfg["render"]["lod"])

            spec = EntitySpec(obj.id, obj.object_type, obj.position, obj.dimension,
                              obj.contour)
            payload, skipped = build_entity_payload(ctx.grid, sorted_xyz, labels,
                                                    [spec], raster_info, camera,
                                                    render, args.patch_grid)
            if not payload:
                records.append({"group": group, "map": map_name, "id": int(obj.id),
                                "object_type": obj.object_type,
                                "skipped": skipped})
                continue
            item = payload[0]
            view, support, crops = item["view"], item["support"], item["crops"]
            shuf = shuffled_correspondence(view, args.patch_grid, seed=int(obj.id) + 3)
            row = {
                "group": group, "map": map_name, "entity_id": int(obj.id),
                "object_type": obj.object_type, "is_target": True,
                "distance_m": distance,
                "support_source": support.source,
                "support_points": len(support),
                "tight_extent_m": entity_scales(obj.dimension)["tight"],
                "context_extent_m": entity_scales(obj.dimension)["context"],
                "td_tight_patches": view.count_of("td", "tight"),
                "td_ctx_patches": view.count_of("td", "context"),
                "o_tight_patches": view.count_of("oblique", "tight"),
                "o_ctx_patches": view.count_of("oblique", "context"),
                "visible_point_ratio": view.visible_point_ratio,
                "correspondence_pairs": int(view.correspondence["weight"].size),
                "coherence": correspondence_coherence(view, args.patch_grid),
                "coherence_shuffled": correspondence_coherence(shuf, args.patch_grid),
            }
            records.append(row)
            fig_row = dict(row)
            fig_row["panels"] = {
                "td_tight": draw_mask_overlay(crops["td"].get("tight", np.zeros((8, 8, 3), np.uint8)),
                                              view.mask_of("td", "tight", args.patch_grid),
                                              MASK_COLOURS["td"]),
                "oblique_tight": draw_mask_overlay(
                    crops["oblique"]["tight"],
                    view.mask_of("oblique", "tight", args.patch_grid),
                    MASK_COLOURS["oblique"]),
                "td_ctx": draw_mask_overlay(crops["td"]["context"],
                                            view.mask_of("td", "context", args.patch_grid),
                                            MASK_COLOURS["td"]),
                "oblique_ctx": draw_mask_overlay(
                    crops["oblique"]["context"],
                    view.mask_of("oblique", "context", args.patch_grid),
                    MASK_COLOURS["oblique"]),
            }
            figure_rows.append(fig_row)
            if len(figure_rows) == 4:
                make_figure(figure_rows,
                            out_dir / f"{group}_{figure_index:02d}.jpg",
                            f"{group}  tight crop (left pair) and context crop (right pair)"
                            f"  patch grid {args.patch_grid}x{args.patch_grid}")
                figure_rows, figure_index = [], figure_index + 1
        if figure_rows:
            make_figure(figure_rows, out_dir / f"{group}_{figure_index:02d}.jpg",
                        f"{group}  tight crop (left pair) and context crop (right pair)"
                        f"  patch grid {args.patch_grid}x{args.patch_grid}")
            figure_rows, figure_index = [], figure_index + 1
            print(f"  {group:9s} {map_name:22s} {obj.object_type:15s} "
                  f"n={len(support):8d} {support.source:20s} "
                  f"o_vis={view.visible_point_ratio * 100:5.1f}% "
                  f"coh={row['coherence']:.2f}/{row['coherence_shuffled']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    summary = {"per_group": {}, "figures": figure_index,
               "totals": {"entities": len([r for r in records if "entity_id" in r])}}
    for group in args.groups:
        rows = [r for r in records if r.get("group") == group and "entity_id" in r]
        if not rows:
            summary["per_group"][group] = {"n": 0}
            continue
        coh = np.array([r["coherence"] for r in rows], float)
        coh_s = np.array([r["coherence_shuffled"] for r in rows], float)
        ok = np.isfinite(coh) & np.isfinite(coh_s)
        sources = defaultdict(int)
        for r in rows:
            sources[r["support_source"]] += 1
        summary["per_group"][group] = {
            "n": len(rows),
            "types": dict(sorted(
                {t: sum(1 for r in rows if r["object_type"] == t)
                 for t in {r["object_type"] for r in rows}}.items())),
            "support_sources": dict(sources),
            "median_support_points": float(np.median([r["support_points"] for r in rows])),
            "median_tight_patches_td": float(np.median(
                [r["td_tight_patches"] for r in rows])),
            "median_ctx_patches_td": float(np.median(
                [r["td_ctx_patches"] for r in rows])),
            "median_o_tight_patches": float(np.median(
                [r["o_tight_patches"] for r in rows])),
            "median_o_ctx_patches": float(np.median(
                [r["o_ctx_patches"] for r in rows])),
            "median_visible_point_ratio": float(np.median(
                [r["visible_point_ratio"] for r in rows])),
            "with_correspondence": int(sum(1 for r in rows
                                           if r["correspondence_pairs"] > 0)),
            "correspondence_success_rate": float(np.mean(
                [r["correspondence_pairs"] > 0 for r in rows])),
            "coherence_real_median": float(np.nanmedian(coh)) if ok.any() else None,
            "coherence_shuffled_median": (float(np.nanmedian(coh_s))
                                          if ok.any() else None),
            "coherence_real_wins": int(np.sum(coh[ok] < coh_s[ok])) if ok.any() else 0,
            "coherence_paired_n": int(ok.sum()),
            "oblique_mask_empty": int(sum(1 for r in rows if r["o_ctx_patches"] == 0)),
            "topdown_mask_empty": int(sum(1 for r in rows if r["td_ctx_patches"] == 0)),
        }

    (out_dir / "entity_projection_check.json").write_text(
        json.dumps({"summary": summary, "records": records},
                   indent=2, sort_keys=True, default=float) + "\n")
    print("\n--- entity projection check ---")
    for group, stats in summary["per_group"].items():
        print(f"  {group}: n={stats.get('n')} types={stats.get('types')}")
        if not stats.get("n"):
            continue
        for key in ("support_sources", "median_support_points",
                    "median_ctx_patches_td", "median_o_ctx_patches",
                    "median_visible_point_ratio", "correspondence_success_rate",
                    "coherence_real_median", "coherence_shuffled_median",
                    "coherence_real_wins"):
            print(f"      {key:30s} {stats[key]}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
