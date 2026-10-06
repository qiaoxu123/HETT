#!/usr/bin/env python3
"""Per-sample geometry, crops and frozen patch features for the fusion round.

One sample is one instruction, one candidate set, one camera.  For each
candidate the script builds the world-coordinate correspondence between the two
views, cuts the two crops on the same ground extent, encodes both with the
frozen SigLIP2 vision tower and writes the patch tokens that the landmark
actually occupies.

What is stored is deliberately *only* what a method needs: the tokens of the
patches inside each landmark mask, the sparse correspondence between them, and
the frozen whole-crop and masked-pooled features.  Storing the full 1024-token
grid for every candidate would be twenty times larger and would let a bug in the
mask go unnoticed, because the mask would no longer be the thing that selected
the tokens.

Discipline for the round: every choice is made on ``train_seen``/``val_seen``.
``val_unseen`` is extracted here and scored once by the trainer, never tuned on.
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
    import cv2  # noqa: F401  imported for the crop writer's side effects
except ImportError:  # pragma: no cover
    cv2 = None

from sensaturban_fpv import citynav  # noqa: E402
from sensaturban_fpv.config import (  # noqa: E402
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.fit_coordinate_transform import (  # noqa: E402
    CoordinateTransform, load_transform,
)
from sensaturban_fpv.entity_geometry import (  # noqa: E402
    candidate_crops, candidate_geometry, crop_extent_m, landmark_point_set,
    oblique_window, project_point,
)
from sensaturban_fpv.pointcloud_renderer import Camera, render_cloud_region  # noqa: E402
from sensaturban_fpv.siglip_masked import (  # noqa: E402
    encode_crops, l2norm, masked_pool,
)
from run_gate_b import (  # noqa: E402
    build_samples, encode_texts, load_encoder, referenced_texts, topdown_raster,
)

OBLIQUE_PITCH = -45.0
OBLIQUE_SIZE = 2048
PATCH_GRID = 32
TEXT_VARIANTS = ("phrase", "name")
# A target whose crop sits far from the frame centre has most of its crop
# outside the image, so the two views would not be covering the same ground.
MAX_CENTRE_OFFSET = 0.34


def build_geo_samples(cfg, objects_by_map, split, target, max_per_map,
                      min_candidates=4, max_candidates=10,
                      same_class_quota=0.6, min_distance=25.0,
                      max_distance=220.0, steps_per_episode=8):
    """Referenced-landmark samples that both views can actually show.

    The candidate-set rule is the one the earlier rounds used -- the referenced
    landmark plus its nearest same-class neighbours and enough other-class
    landmarks to fill the set -- so the task stays the same task.  The new
    constraint is on the *pose*: the referenced landmark has to project into the
    middle of the oblique frame, otherwise the crop it needs is mostly outside
    the image and the two views are not looking at the same ground.
    """
    episodes = citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split)
    xyz_of = {
        map_name: {oid: tuple(float(v) for v in o.position)
                   for oid, o in objects.items()}
        for map_name, objects in objects_by_map.items()
    }

    def dist2(map_name, a_id, b_id):
        ax, ay, az = xyz_of[map_name][a_id]
        bx, by, bz = xyz_of[map_name][b_id]
        return (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2

    pitch = np.deg2rad(OBLIQUE_PITCH)
    samples, per_map = [], defaultdict(int)
    for episode in episodes:
        if per_map[episode.map_name] >= max_per_map:
            continue
        objects = objects_by_map.get(episode.map_name, {})
        if not episode.object_ids or not objects:
            continue
        ref = objects.get(episode.object_ids[0])
        if ref is None:
            continue
        texts = referenced_texts(ref)
        if not any(texts.values()):
            continue
        others = [o for oid, o in objects.items() if oid != ref.id]
        if len(others) < min_candidates - 1:
            continue

        position = episode.trajectory[:, :3]
        yaws = episode.yaw()
        found = None
        for step in np.linspace(0, len(position) - 1,
                                num=min(steps_per_episode, len(position))).astype(int):
            pos = position[step]
            yaw = float(yaws[step])
            distance = float(np.linalg.norm(
                np.asarray(ref.position, dtype=np.float64) - pos))
            if not (min_distance <= distance <= max_distance):
                continue
            cam = Camera(position=pos, yaw=yaw, pitch=pitch,
                         width=OBLIQUE_SIZE, height=OBLIQUE_SIZE,
                         hfov_deg=cfg["render"]["hfov_deg"],
                         near=cfg["render"]["near"], far=cfg["render"]["far"])
            proj = project_point(np.asarray(ref.position, float), cam)
            if proj is None:
                continue
            du = abs(proj["u"] - OBLIQUE_SIZE / 2.0) / OBLIQUE_SIZE
            dv = abs(proj["v"] - OBLIQUE_SIZE / 2.0) / OBLIQUE_SIZE
            if max(du, dv) > MAX_CENTRE_OFFSET:
                continue
            found = (int(step), pos, yaw, distance, proj)
            break
        if found is None:
            continue
        step, pos, yaw, distance, proj = found

        same_class = [o for o in others if o.object_type == ref.object_type]
        other_class = [o for o in others if o.object_type != ref.object_type]
        n_same = int(round(max_candidates * same_class_quota))
        same_class.sort(key=lambda o: dist2(episode.map_name, o.id, ref.id))
        other_class.sort(key=lambda o: dist2(episode.map_name, o.id, ref.id))
        candidates = ([ref] + same_class[:n_same] + other_class)[:max_candidates]
        if len(candidates) < min_candidates:
            continue
        per_map[episode.map_name] += 1

        samples.append({
            "split": split, "map": episode.map_name,
            "episode_index": int(episode.index), "step": step,
            "position": np.asarray(pos, dtype=np.float64),
            "yaw": yaw, "instruction": episode.description, "texts": texts,
            "target_id": int(ref.id), "target_name": ref.name,
            "target_type": ref.object_type, "target_distance": distance,
            "target_centre_offset": float(max(du, dv)),
            "candidate_ids": [int(o.id) for o in candidates],
            "candidate_types": [o.object_type for o in candidates],
            "same_class_candidates": int(sum(
                1 for o in candidates[1:] if o.object_type == ref.object_type)),
        })
        if len(samples) >= target:
            break
    return samples


def candidate_payload(grid, sorted_xyz, objs, sample, raster_info, camera,
                      render_result, patch_grid):
    """Everything the round needs about one candidate, or why it was skipped."""
    payload, skipped = [], defaultdict(int)
    for cid in sample["candidate_ids"]:
        obj = objs[cid]
        # The footprint filter is applied inside candidate_geometry, so the raw
        # box query is what goes in and the polygon is applied exactly once.
        pts = landmark_point_set(grid, sorted_xyz, obj.position, obj.dimension)
        geom = candidate_geometry(cid, obj.position, obj.dimension, pts,
                                  raster_info, camera, render_result, sorted_xyz,
                                  patch_grid=patch_grid, footprint=obj.contour)
        if geom is None:
            skipped["no_geometry"] += 1
            continue
        views = candidate_crops(raster_info, camera, render_result.rgb,
                                obj.position, geom.extent_m)
        if views["td"] is None:
            skipped["no_topdown_crop"] += 1
            continue
        if views["oblique"] is None:
            skipped["no_oblique_crop"] += 1
            continue
        if geom.td_patch_count == 0:
            skipped["empty_topdown_mask"] += 1
            continue
        if geom.o_patch_count == 0:
            skipped["empty_oblique_mask"] += 1
            continue
        payload.append({"id": int(cid), "obj": obj, "geom": geom, "views": views,
                        "is_target": bool(cid == sample["target_id"])})
    return payload, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits", nargs="*",
                    default=["train_seen", "val_seen", "val_unseen"])
    ap.add_argument("--per-split", type=int, default=700)
    ap.add_argument("--max-per-map", type=int, default=80)
    ap.add_argument("--per-split-spec", default=None,
                    help="'split:target:max_per_map,...' overriding the two above. "
                         "val_unseen has only four blocks, so it needs a higher "
                         "per-map cap than the training splits to reach the same n.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "geo_fusion" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform_path = Path(args.transform) if args.transform else \
        artifact_dir(cfg) / "transform" / "best_transform.json"
    transform = load_transform(transform_path) if transform_path.exists() \
        else CoordinateTransform.identity()

    objects_by_map = load_landmarks(cfg)
    processor, model, torch = load_encoder(cfg, args.device)
    print(f"encoder {cfg['vision_probe']['model']} on {args.device}", flush=True)

    spec = {}
    if args.per_split_spec:
        for chunk in args.per_split_spec.split(","):
            name, target, cap = chunk.split(":")
            spec[name.strip()] = (int(target), int(cap))

    for split in args.splits:
        per_split, max_per_map = spec.get(split, (args.per_split, args.max_per_map))
        samples = build_geo_samples(cfg, objects_by_map, split, per_split, max_per_map)
        if args.limit:
            samples = samples[: args.limit]
        shard_path = out_dir / f"{split}.shard{args.shard}.jsonl"
        done = set()
        for path in sorted(out_dir.glob(f"{split}*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("kept"):
                        done.add((rec["episode_index"], rec["step"]))
        print(f"{split}: {len(samples)} samples built, {len(done)} kept already",
              flush=True)

        unique_texts = sorted({s["texts"][v] for s in samples for v in TEXT_VARIANTS
                               if s["texts"].get(v)})
        text_feats = encode_texts(processor, model, torch, unique_texts, args.device)
        text_index = {t: i for i, t in enumerate(unique_texts)}

        cache, rasters = {}, {}
        stats = defaultdict(int)
        t0 = time.time()
        for n, sample in enumerate(samples):
            if (sample["episode_index"], sample["step"]) in done:
                continue
            if args.shards > 1 and (n % args.shards) != args.shard:
                continue
            map_name = sample["map"]
            if map_name not in cache:
                cache = {map_name: build_map_context(cfg, map_name,
                                                     objects_by_map=objects_by_map)}
            ctx = cache[map_name]
            if map_name not in rasters:
                rasters[map_name] = topdown_raster(Path(cfg["paths"]["ortho_dir"]),
                                                   map_name)
            raster_info = rasters[map_name]
            sorted_xyz, _ = ctx.grid.sorted_arrays()
            objs = objects_by_map[map_name]

            cam = Camera(position=sample["position"], yaw=sample["yaw"],
                         pitch=np.deg2rad(OBLIQUE_PITCH),
                         width=OBLIQUE_SIZE, height=OBLIQUE_SIZE,
                         hfov_deg=cfg["render"]["hfov_deg"],
                         near=cfg["render"]["near"], far=cfg["render"]["far"])
            t_render = time.time()
            render = render_cloud_region(ctx.cloud, ctx.grid, cam, splat_radius=0,
                                         lod=cfg["render"]["lod"])
            render_s = time.time() - t_render

            payload, skipped = candidate_payload(ctx.grid, sorted_xyz, objs, sample,
                                                 raster_info, cam, render, PATCH_GRID)
            for key, value in skipped.items():
                stats[key] += value

            target_rows = [i for i, p in enumerate(payload) if p["is_target"]]
            usable = target_rows and payload[target_rows[0]]["geom"].corr_td.size > 0
            if not usable:
                stats["target_without_correspondence"] += 1
                record = {"split": split, "map": map_name, "kept": False,
                          "episode_index": int(sample["episode_index"]),
                          "step": int(sample["step"]),
                          "reason": ("target_missing" if not target_rows
                                     else "target_without_correspondence"),
                          "candidates": len(payload)}
                with shard_path.open("a") as handle:
                    handle.write(json.dumps(record, default=float) + "\n")
                continue

            t_enc = time.time()
            td_tokens, td_pooled, grid_p = encode_crops(
                processor, model, [p["views"]["td"] for p in payload], args.device)
            o_tokens, o_pooled, grid_o = encode_crops(
                processor, model, [p["views"]["oblique"] for p in payload], args.device)
            if grid_p != PATCH_GRID or grid_o != PATCH_GRID:
                raise ValueError(f"unexpected patch grid {grid_p}/{grid_o}")
            encode_s = time.time() - t_enc

            # Frozen masked pooling: the same head the whole-crop feature comes
            # from, attending only to the landmark's own patches.
            td_keep = torch.from_numpy(np.stack(
                [p["geom"].td_patches.reshape(-1) for p in payload]))
            o_keep = torch.from_numpy(np.stack(
                [p["geom"].o_patches.reshape(-1) for p in payload]))
            with torch.no_grad():
                td_masked = masked_pool(model, td_tokens.to(args.device), td_keep)
                o_masked = masked_pool(model, o_tokens.to(args.device), o_keep)

            # Compact per-candidate token blocks: only the patches the masks
            # select, so the mask is what decides what a method can see.
            td_tok, td_row, td_patch = [], [], []
            o_tok, o_row, o_patch = [], [], []
            corr_cand, corr_td_row, corr_o_row, corr_w = [], [], [], []
            patch_of = {}
            for i, p in enumerate(payload):
                geom = p["geom"]
                flat_td = np.flatnonzero(geom.td_patches.reshape(-1))
                flat_o = np.flatnonzero(geom.o_patches.reshape(-1))
                patch_of[i] = (
                    {int(v): j for j, v in enumerate(flat_td.tolist())},
                    {int(v): j for j, v in enumerate(flat_o.tolist())},
                )
                td_tok.append(td_tokens[i][flat_td])
                td_row.append(np.full(len(flat_td), i, dtype=np.int32))
                td_patch.append(flat_td.astype(np.int32))
                o_tok.append(o_tokens[i][flat_o])
                o_row.append(np.full(len(flat_o), i, dtype=np.int32))
                o_patch.append(flat_o.astype(np.int32))

            # Offsets turn the per-candidate patch lists into one flat array.
            td_off = np.concatenate([[0], np.cumsum([len(x) for x in td_patch])])
            o_off = np.concatenate([[0], np.cumsum([len(x) for x in o_patch])])
            for i, p in enumerate(payload):
                geom = p["geom"]
                if geom.corr_td.size == 0:
                    continue
                td_map, o_map = patch_of[i]
                td_local = np.array([td_map[int(v)] for v in geom.corr_td], np.int64)
                o_local = np.array([o_map[int(v)] for v in geom.corr_o], np.int64)
                corr_cand.append(np.full(len(td_local), i, dtype=np.int32))
                corr_td_row.append((td_local + td_off[i]).astype(np.int32))
                corr_o_row.append((o_local + o_off[i]).astype(np.int32))
                corr_w.append(geom.corr_weight.astype(np.float32))

            # Text features travel in the archive, not in the JSON record: an
            # ndarray would make the record unserialisable, and the encoder is
            # deliberately not re-run at training time.
            texts_out = {v: sample["texts"][v] for v in TEXT_VARIANTS
                         if sample["texts"].get(v)}

            arrays = {
                "patch_grid": np.int32(PATCH_GRID),
                "candidate_ids": np.array([p["id"] for p in payload], np.int64),
                "candidate_types": np.array([p["obj"].object_type for p in payload]),
                "is_target": np.array([p["is_target"] for p in payload], bool),
                "extent_m": np.array([p["geom"].extent_m for p in payload], np.float32),
                "td_patch_count": np.array([p["geom"].td_patch_count for p in payload],
                                           np.int32),
                "o_patch_count": np.array([p["geom"].o_patch_count for p in payload],
                                          np.int32),
                "o_visible_points": np.array(
                    [p["geom"].o_visible_point_count for p in payload], np.int64),
                "td_tokens": torch.cat(td_tok).to(torch.float16).numpy(),
                "td_row": np.concatenate(td_row),
                "td_patch": np.concatenate(td_patch),
                "o_tokens": torch.cat(o_tok).to(torch.float16).numpy(),
                "o_row": np.concatenate(o_row),
                "o_patch": np.concatenate(o_patch),
                # All four land in the same space and are L2-normalised here, so
                # every method's score is a cosine and the stored features are
                # directly comparable across methods and across samples.
                "td_global": l2norm(td_pooled).numpy().astype(np.float32),
                "o_global": l2norm(o_pooled).numpy().astype(np.float32),
                "td_masked": l2norm(td_masked).float().cpu().numpy(),
                "o_masked": l2norm(o_masked).float().cpu().numpy(),
                "corr_cand": (np.concatenate(corr_cand) if corr_cand
                              else np.zeros(0, np.int32)),
                "corr_td_row": (np.concatenate(corr_td_row) if corr_td_row
                                else np.zeros(0, np.int32)),
                "corr_o_row": (np.concatenate(corr_o_row) if corr_o_row
                               else np.zeros(0, np.int32)),
                "corr_w": (np.concatenate(corr_w) if corr_w
                           else np.zeros(0, np.float32)),
            }
            for variant in TEXT_VARIANTS:
                text = sample["texts"].get(variant)
                arrays[f"text_{variant}"] = (
                    text_feats[text_index[text]].detach().cpu().numpy().astype(np.float32)
                    if text else np.zeros(0, np.float32))
            stem = f"{split}_{sample['episode_index']:05d}_step{sample['step']:03d}"
            np.savez(out_dir / f"{stem}.npz", **arrays)

            record = {
                "split": split, "map": map_name, "kept": True,
                "episode_index": int(sample["episode_index"]),
                "step": int(sample["step"]), "file": f"{stem}.npz",
                "target_id": int(sample["target_id"]),
                "target_type": sample["target_type"],
                "target_distance": float(sample["target_distance"]),
                "target_centre_offset": float(sample["target_centre_offset"]),
                "same_class_candidates": int(sample["same_class_candidates"]),
                "instruction": sample["instruction"], "texts": texts_out,
                "candidate_ids": [p["id"] for p in payload],
                "candidate_types": [p["obj"].object_type for p in payload],
                "candidate_count": len(payload),
                "correspondence_pairs": int(sum(
                    p["geom"].corr_td.size for p in payload)),
                "target_correspondence_pairs": int(
                    payload[target_rows[0]]["geom"].corr_td.size),
                "target_crop_extent_m": float(payload[target_rows[0]]["geom"].extent_m),
                "oblique_valid_pixel_ratio": float(render.valid_ratio),
                "oblique_in_tile_valid_ratio": render.stats.get("in_tile_valid_ratio"),
                "render_seconds": round(render_s, 3),
                "encode_seconds": round(encode_s, 3),
                "skipped": dict(skipped),
            }
            with shard_path.open("a") as handle:
                handle.write(json.dumps(record, default=float) + "\n")
            stats["kept"] += 1
            if (n + 1) % 10 == 0:
                print(f"  {split} {n + 1}/{len(samples)} kept={stats['kept']} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        print(f"{split} shard {args.shard}: {dict(stats)}", flush=True)
        summary_path = out_dir / f"{split}.shard{args.shard}.summary.json"
        summary_path.write_text(json.dumps({"split": split, "shard": args.shard,
                                            "shards": args.shards,
                                            "stats": dict(stats)},
                                           indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
