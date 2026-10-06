#!/usr/bin/env python3
"""Candidate-level scores for the dual-view fusion round.

Produces, for three splits under one identical recipe:

* a top-down crop per candidate at 0.10 / 0.15 / 0.20 / 0.30 m/px
* an oblique45 crop per candidate at 2048 px (the best perspective setting)
* SigLIP2 similarity against both the referenced phrase and the landmark name
* per-sample quality features for the quality-aware gate

Discipline for the round: ``val_unseen`` is test only.  Every choice -- the
top-down operating point, the calibration, the fusion weights, the gate
thresholds -- is made on ``train_seen`` / ``val_seen``, and val_unseen is scored
once at the end.

Every source uses the same physical crop extent for a given landmark, so the
landmark's share of the crop is equal by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from sensaturban_fpv import citynav  # noqa: E402
from sensaturban_fpv.config import (  # noqa: E402
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.fit_coordinate_transform import (  # noqa: E402
    CoordinateTransform, load_transform,
)
from sensaturban_fpv.pointcloud_renderer import Camera, render_cloud_region  # noqa: E402
from sensaturban_fpv.project_landmarks import project_centre  # noqa: E402

from run_gate_b import (  # noqa: E402
    build_samples, crop_square, encode_images, encode_texts, load_encoder,
    topdown_raster,
)
from run_resolution_control import crop_extent_m, degrade, landmark_pixel_ratio  # noqa: E402

OBLIQUE_PITCH = -45.0
OBLIQUE_SIZE = 2048
TD_RESOLUTIONS = (0.10, 0.15, 0.20, 0.30)
TEXT_VARIANTS = ("phrase", "name")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits", nargs="*", default=["val_seen", "train_seen", "val_unseen"])
    ap.add_argument("--per-split", type=int, default=150)
    ap.add_argument("--max-per-map", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "fusion" / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform_path = Path(args.transform) if args.transform else \
        artifact_dir(cfg) / "transform" / "best_transform.json"
    transform = load_transform(transform_path) if transform_path.exists() \
        else CoordinateTransform.identity()

    objects_by_map = load_landmarks(cfg)
    processor, model, torch = load_encoder(cfg, args.device)
    print(f"encoder on {args.device}", flush=True)

    for split in args.splits:
        samples = build_samples(cfg, objects_by_map, [split], args.per_split,
                                max_per_map=args.max_per_map)
        already = set()
        split_paths = sorted(out_dir.glob(f"{split}*.jsonl"))
        for path in split_paths:
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    already.add((rec["episode_index"], rec["step"]))
        path_out = out_dir / (f"{split}.jsonl" if args.shards == 1
                              else f"{split}.shard{args.shard}.jsonl")
        print(f"{split}: {len(samples)} samples built, {len(already)} already done",
              flush=True)

        splits_cache = {split: citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split)}
        episode_of = {e.index: e for e in splits_cache[split]}

        unique_texts = sorted({s["texts"][v] for s in samples for v in TEXT_VARIANTS
                               if s["texts"].get(v)})
        text_feats = encode_texts(processor, model, torch, unique_texts, args.device)
        text_index = {t: i for i, t in enumerate(unique_texts)}

        cache, rasters = {}, {}
        t0 = time.time()
        for n, sample in enumerate(samples):
            if (sample["episode_index"], sample["step"]) in already:
                continue
            if args.shards > 1 and (n % args.shards) != args.shard:
                continue
            map_name = sample["map"]
            if map_name not in cache:
                cache = {map_name: build_map_context(cfg, map_name,
                                                     objects_by_map=objects_by_map)}
            ctx = cache[map_name]
            if map_name not in rasters:
                rasters[map_name] = topdown_raster(Path(cfg["paths"]["ortho_dir"]), map_name)
            raster_info = rasters[map_name]

            episode = episode_of[sample["episode_index"]]
            position = transform.apply_xyz(
                np.asarray(episode.trajectory[sample["step"], :3])[None, :])[0]
            yaw = float(episode.yaw()[sample["step"]])

            cam = Camera(position=position, yaw=yaw, pitch=np.deg2rad(OBLIQUE_PITCH),
                         width=OBLIQUE_SIZE, height=OBLIQUE_SIZE,
                         hfov_deg=cfg["render"]["hfov_deg"],
                         near=cfg["render"]["near"], far=cfg["render"]["far"])
            t_render = time.time()
            render = render_cloud_region(ctx.cloud, ctx.grid, cam, splat_radius=0,
                                         lod=cfg["render"]["lod"])
            render_s = time.time() - t_render

            objs = objects_by_map[map_name]
            cand_objs = [objs[c] for c in sample["candidate_ids"]]
            sources = {}
            for cobj in cand_objs:
                pos = np.asarray(cobj.position, dtype=np.float64)
                extent = crop_extent_m(cobj.dimension)
                entry = {}
                for res in TD_RESOLUTIONS:
                    key = f"td_{int(round(res * 100)):03d}"
                    if raster_info is None:
                        entry[key] = None
                        continue
                    raster, origin, native = raster_info
                    col = (pos[0] - origin[0]) / native
                    rowp = (origin[1] - pos[1]) / native
                    if not (0 <= rowp < raster.shape[0] and 0 <= col < raster.shape[1]):
                        entry[key] = None
                        continue
                    img = crop_square(raster, col, rowp, extent / native)
                    entry[key] = degrade(img, res, native)
                proj = project_centre(pos, cam)
                if proj is None or not proj["inside"]:
                    entry["o2048"] = None
                else:
                    side_px = cam.focal * extent / max(proj["metric"], 1e-3)
                    entry["o2048"] = crop_square(render.rgb, proj["u"], proj["v"], side_px)
                sources[cobj.id] = entry

            target = objs[sample["target_id"]]
            target_lm = (target.id, target.name, target.object_type,
                         tuple(float(v) for v in target.position),
                         tuple(float(v) for v in target.dimension))
            sorted_xyz, _ = ctx.grid.sorted_arrays()

            record = {
                "split": split, "map": map_name,
                "episode_index": int(sample["episode_index"]), "step": int(sample["step"]),
                "instruction": sample["instruction"], "texts": sample["texts"],
                "target_id": int(sample["target_id"]),
                "target_type": sample["target_type"],
                "target_distance": float(sample["target_distance"]),
                "same_class_candidates": int(sample["same_class_candidates"]),
                "candidate_ids": [int(c) for c in sample["candidate_ids"]],
                "candidate_types": list(sample["candidate_types"]),
                "candidate_dimensions": list(sample["candidate_dimensions"]),
                "crop_extent_m": {o.id: crop_extent_m(o.dimension) for o in cand_objs},
                "quality": {
                    "oblique_valid_pixel_ratio": float(render.valid_ratio),
                    "oblique_in_tile_valid_ratio": render.stats.get("in_tile_valid_ratio"),
                    "target_pixel_ratio": landmark_pixel_ratio(
                        render, cam, target_lm, ctx.grid, sorted_xyz),
                    "target_points_projected": int(render.stats["points_projected"]),
                    "target_in_frame": bool(
                        (lambda p: p is not None and p["inside"])(
                            project_centre(np.asarray(target.position, dtype=np.float64), cam))),
                    "render_seconds": round(render_s, 3),
                },
                "views": {},
            }

            for name in [f"td_{int(round(r * 100)):03d}" for r in TD_RESOLUTIONS] + ["o2048"]:
                ids = [c for c in sample["candidate_ids"] if sources[c].get(name) is not None]
                if sample["target_id"] not in ids or len(ids) < 2:
                    continue
                images = [sources[c][name] for c in ids]
                if cv2 is not None:
                    d = out_dir / "crops" / (
                        f"{split}_{sample['episode_index']:05d}_step{sample['step']:03d}")
                    d.mkdir(parents=True, exist_ok=True)
                    for c in ids:
                        cv2.imwrite(str(d / f"{name}_cand{c}.png"),
                                    cv2.cvtColor(sources[c][name], cv2.COLOR_RGB2BGR))
                feats = encode_images(processor, model, torch, images, args.device)
                sims = {}
                for variant in TEXT_VARIANTS:
                    text = sample["texts"].get(variant)
                    if not text:
                        continue
                    tf = text_feats[text_index[text]:text_index[text] + 1]
                    sims[variant] = [float(x) for x in
                                     (feats @ tf.T).squeeze(-1).cpu().numpy()]
                record["views"][name] = {"ids": [int(c) for c in ids], "sims": sims}

            with path_out.open("a") as handle:
                handle.write(json.dumps(record, default=float) + "\n")
            if (n + 1) % 10 == 0:
                print(f"  {split} {n + 1}/{len(samples)} ({time.time() - t0:.0f}s)",
                      flush=True)
        print(f"{split} done -> {path_out}", flush=True)


if __name__ == "__main__":
    main()
