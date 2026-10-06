#!/usr/bin/env python3
"""Per-sample 3D anchors, multi-scale crops and frozen patch features.

One sample is one instruction, one candidate set, one camera.  For every
candidate entity -- of any type, sized by its own annotation -- the script builds
its 3D support, projects that support into the top-down raster and the oblique
frame at two scales, cuts the four crops, encodes them with the frozen SigLIP2
vision tower, and writes the patch tokens the entity's masks select.

What is written per sample:

* tight- and context-scale masks, per view, per scale;
* the tokens of the context-scale patches each entity occupies, in each view,
  and the sparse correspondence between them;
* frozen pooled features: entity-masked and background-masked, per view, per
  scale, plus the whole-crop features the earlier rounds scored.

The whole-crop features are kept so this round can be read against the previous
ones on the same samples, but nothing here is restricted to a view or a scale:
the point is that a method may consume any of them.

Discipline: every choice is made on ``train_seen`` / ``val_seen``, and
``val_unseen`` is written here and scored once by the trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv import citynav  # noqa: E402
from sensaturban_fpv.config import (  # noqa: E402
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.entity_geometry import (  # noqa: E402
    EntitySpec, build_entity_payload, entity_scales, group_of, project_point,
)
from sensaturban_fpv.pointcloud_renderer import Camera, render_cloud_region  # noqa: E402
from sensaturban_fpv.siglip_masked import (  # noqa: E402
    encode_crops, l2norm, masked_pool,
)
from run_gate_b import encode_texts, load_encoder, referenced_texts, topdown_raster  # noqa: E402

OBLIQUE_PITCH = -45.0
OBLIQUE_SIZE = 2048
PATCH_GRID = 32
TEXT_VARIANTS = ("phrase", "name")
SCALES = ("tight", "context")
# A target whose centre sits near the frame edge has most of its context crop
# outside the image, so the two views would not cover the same ground.
MAX_CENTRE_OFFSET = 0.30


def build_entity_samples(cfg, objects_by_map, split, target, max_per_map,
                         min_candidates=4, max_candidates=10,
                         same_class_quota=0.6, min_distance=25.0,
                         max_distance=220.0, steps_per_episode=8,
                         group_quotas=None):
    """Instruction samples whose referenced entity both views can show.

    The candidate-set rule is the one the earlier rounds used -- the referenced
    entity plus its nearest same-class neighbours and enough other-class
    entities to fill the set -- so the task stays the same task.  It is written
    in terms of entity types rather than any particular type: nothing here
    prefers a building.

    ``group_quotas`` caps how many samples each type group may contribute.
    Without it the set simply follows file order, and on ``train_seen`` that
    order is 91% buildings -- a fusion trained there would be measured on cars
    it had almost never seen, which is a distribution shift rather than a result.
    """
    episodes = citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split)
    xyz_of = {m: {oid: tuple(float(v) for v in o.position) for oid, o in objs.items()}
              for m, objs in objects_by_map.items()}

    def dist2(map_name, a_id, b_id):
        ax, ay, az = xyz_of[map_name][a_id]
        bx, by, bz = xyz_of[map_name][b_id]
        return (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2

    pitch = np.deg2rad(OBLIQUE_PITCH)
    samples, per_map = [], defaultdict(int)
    taken_by_group = Counter()
    for episode in episodes:
        if per_map[episode.map_name] >= max_per_map:
            continue
        objects = objects_by_map.get(episode.map_name, {})
        if not episode.object_ids or not objects:
            continue
        ref = objects.get(episode.object_ids[0])
        if ref is None:
            continue
        group = group_of(ref.object_type)
        if group_quotas is not None and taken_by_group[group] >= group_quotas.get(group, 0):
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
            distance = float(np.linalg.norm(
                np.asarray(ref.position, dtype=np.float64) - pos))
            if not (min_distance <= distance <= max_distance):
                continue
            cam = Camera(position=pos, yaw=float(yaws[step]), pitch=pitch,
                         width=OBLIQUE_SIZE, height=OBLIQUE_SIZE,
                         hfov_deg=cfg["render"]["hfov_deg"],
                         near=cfg["render"]["near"], far=cfg["render"]["far"])
            proj = project_point(np.asarray(ref.position, float), cam)
            if proj is None:
                continue
            offset = max(abs(proj["u"] - OBLIQUE_SIZE / 2.0),
                         abs(proj["v"] - OBLIQUE_SIZE / 2.0)) / OBLIQUE_SIZE
            if offset > MAX_CENTRE_OFFSET:
                continue
            found = (int(step), pos, float(yaws[step]), distance, offset)
            break
        if found is None:
            continue
        step, pos, yaw, distance, offset = found

        same_class = [o for o in others if o.object_type == ref.object_type]
        other_class = [o for o in others if o.object_type != ref.object_type]
        n_same = int(round(max_candidates * same_class_quota))
        same_class.sort(key=lambda o: dist2(episode.map_name, o.id, ref.id))
        other_class.sort(key=lambda o: dist2(episode.map_name, o.id, ref.id))
        candidates = ([ref] + same_class[:n_same] + other_class)[:max_candidates]
        if len(candidates) < min_candidates:
            continue
        per_map[episode.map_name] += 1
        taken_by_group[group] += 1

        samples.append({
            "split": split, "map": episode.map_name,
            "episode_index": int(episode.index), "step": step,
            "position": np.asarray(pos, dtype=np.float64), "yaw": yaw,
            "instruction": episode.description, "texts": texts,
            "target_id": int(ref.id), "target_name": ref.name,
            "target_type": ref.object_type, "target_distance": distance,
            "target_centre_offset": float(offset),
            "candidate_ids": [int(o.id) for o in candidates],
            "candidate_types": [o.object_type for o in candidates],
            "same_class_candidates": int(sum(
                1 for o in candidates[1:] if o.object_type == ref.object_type)),
            "target_group": group,
        })
        if len(samples) >= target:
            break
    return samples


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
                    help="'split:target:max_per_map,...' overriding the two above")
    ap.add_argument("--group-quotas", default="building:450,vehicle:350,other:200",
                    help="cap per type group; 'none' to follow file order instead")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    out_dir = Path(args.out) if args.out else \
        artifact_dir(cfg) / "entity_grounding" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_by_map = load_landmarks(cfg)
    processor, model, torch = load_encoder(cfg, args.device)
    print(f"encoder {cfg['vision_probe']['model']} on {args.device}", flush=True)

    spec = {}
    if args.per_split_spec:
        for chunk in args.per_split_spec.split(","):
            name, target, cap = chunk.split(":")
            spec[name.strip()] = (int(target), int(cap))

    quotas = None
    if args.group_quotas and args.group_quotas != "none":
        quotas = {k: int(v) for k, v in
                  (chunk.split(":") for chunk in args.group_quotas.split(","))}
        total = sum(quotas.values())
        print(f"group quotas {quotas} (sum {total}); targets are scaled to match",
              flush=True)

    for split in args.splits:
        per_split, max_per_map = spec.get(split, (args.per_split, args.max_per_map))
        if quotas:
            scale = per_split / sum(quotas.values())
            split_quotas = {k: max(int(round(v * scale)), 1)
                            for k, v in quotas.items()}
        else:
            split_quotas = None
        samples = build_entity_samples(cfg, objects_by_map, split, per_split,
                                       max_per_map, group_quotas=split_quotas)
        if args.limit:
            samples = samples[: args.limit]
        shard_path = out_dir / f"{split}.shard{args.shard}.jsonl"
        done = set()
        for path in sorted(out_dir.glob(f"{split}*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("kept"):
                        done.add((record["episode_index"], record["step"]))
        print(f"{split}: {len(samples)} samples built, {len(done)} kept already",
              flush=True)

        unique_texts = sorted({s["texts"][v] for s in samples for v in TEXT_VARIANTS
                               if s["texts"].get(v)})
        text_feats = encode_texts(processor, model, torch, unique_texts, args.device)
        text_index = {t: i for i, t in enumerate(unique_texts)}

        cache, rasters = {}, {}
        stats = Counter()
        type_counts = Counter()
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
            labels = ctx.grid.sorted_labels()
            objs = objects_by_map[map_name]

            camera = Camera(position=sample["position"], yaw=sample["yaw"],
                            pitch=np.deg2rad(OBLIQUE_PITCH),
                            width=OBLIQUE_SIZE, height=OBLIQUE_SIZE,
                            hfov_deg=cfg["render"]["hfov_deg"],
                            near=cfg["render"]["near"], far=cfg["render"]["far"])
            t_render = time.time()
            render = render_cloud_region(ctx.cloud, ctx.grid, camera, splat_radius=0,
                                         lod=cfg["render"]["lod"])
            render_s = time.time() - t_render

            specs = [EntitySpec(cid, objs[cid].object_type, objs[cid].position,
                                objs[cid].dimension, objs[cid].contour)
                     for cid in sample["candidate_ids"]]
            payload, skipped = build_entity_payload(ctx.grid, sorted_xyz, labels,
                                                    specs, raster_info, camera,
                                                    render, PATCH_GRID)
            for key, value in skipped.items():
                stats[key] += value

            target_rows = [i for i, p in enumerate(payload)
                           if p["spec"].entity_id == sample["target_id"]]
            if not target_rows:
                stats["target_dropped"] += 1
                reason = "target_dropped"
            elif payload[target_rows[0]]["view"].scales["oblique"].get("context") is None:
                stats["target_not_in_oblique_frame"] += 1
                reason = "target_not_in_oblique_frame"
            else:
                reason = None
            if reason:
                with shard_path.open("a") as handle:
                    handle.write(json.dumps({
                        "split": split, "map": map_name, "kept": False,
                        "episode_index": int(sample["episode_index"]),
                        "step": int(sample["step"]), "reason": reason,
                    }) + "\n")
                continue

            t_enc = time.time()
            panels, pooled, grids = {}, {}, set()
            for view in ("td", "oblique"):
                for scale in SCALES:
                    images = [p["crops"][view].get(scale) for p in payload]
                    usable = [img for img in images if img is not None]
                    if len(usable) != len(images):
                        raise ValueError(f"missing {view}/{scale} crop for a candidate")
                    tokens, pooled_feat, grid = encode_crops(processor, model,
                                                             images, args.device)
                    panels[(view, scale)] = tokens
                    pooled[(view, scale)] = pooled_feat
                    grids.add(grid)
            if grids != {PATCH_GRID}:
                raise ValueError(f"unexpected patch grids {grids}")
            encode_s = time.time() - t_enc

            # Frozen masked pooling: the head the whole-crop feature comes from,
            # attending only to the patches the entity occupies -- and, for the
            # background feature, only to the ones it does not.
            masked = {}
            with torch.no_grad():
                for (view, scale), tokens in panels.items():
                    keep = torch.from_numpy(np.stack([
                        p["view"].mask_of(view, scale, PATCH_GRID).reshape(-1)
                        for p in payload]))
                    masked[(view, scale, "entity")] = l2norm(
                        masked_pool(model, tokens.to(args.device), keep)
                    ).float().cpu()
                    bg = ~keep
                    any_bg = bg.any(dim=1)
                    masked[(view, scale, "background")] = l2norm(
                        masked_pool(model, tokens.to(args.device),
                                    torch.where(any_bg[:, None], bg, keep))
                    ).float().cpu()

                # Patches only this view resolved.  Both views' masks come from
                # the same 3D support, so a patch is exclusive exactly when the
                # other view never saw those points -- a facade, a roof, or an
                # occluded side.  Dropping them would discard the only evidence
                # there is for whatever the other view cannot show.  A candidate
                # with none falls back to its full entity mask, which is the
                # honest reading of "no exclusive evidence" rather than an empty
                # pooling over nothing.
                for view in ("td", "oblique"):
                    stored = panels[(view, "context")]
                    width = max(stored[i].shape[0] for i in range(len(payload)))
                    keep_excl = torch.zeros((len(payload), width), dtype=torch.bool)
                    counts = []
                    for i, p in enumerate(payload):
                        flat = np.flatnonzero(
                            p["view"].mask_of(view, "context", PATCH_GRID).reshape(-1))
                        corr = p["view"].correspondence
                        # The correspondence is one table per entity, keyed by
                        # view side ("td" / "o"), not by view name.
                        side = "td" if view == "td" else "o"
                        shared = ({int(v) for v in corr[side]}
                                  if corr["weight"].size else set())
                        sel = np.array([j for j, f in enumerate(flat.tolist())
                                        if int(f) not in shared], dtype=np.int64)
                        counts.append(int(sel.size))
                        if sel.size:
                            keep_excl[i, :sel.size] = torch.from_numpy(sel)
                    keep = torch.from_numpy(np.stack([
                        p["view"].mask_of(view, "context", PATCH_GRID).reshape(-1)
                        for p in payload]))
                    any_excl = keep_excl.any(dim=1)
                    masked[(view, "context", "exclusive")] = l2norm(
                        masked_pool(model, stored.to(args.device),
                                    torch.where(any_excl[:, None], keep_excl, keep))
                    ).float().cpu()
                    masked[(view, "exclusive_count")] = np.array(counts, np.int32)

            # Compact token blocks: only the context-scale patches each entity's
            # mask selects, so the mask is what decides what a method can see.
            arrays = {"patch_grid": np.int32(PATCH_GRID)}
            for view in ("td", "oblique"):
                token_rows, token_row_id, token_patch = [], [], []
                for i, p in enumerate(payload):
                    flat = np.flatnonzero(
                        p["view"].mask_of(view, "context", PATCH_GRID).reshape(-1))
                    token_rows.append(panels[(view, "context")][i][flat])
                    token_row_id.append(np.full(len(flat), i, dtype=np.int32))
                    token_patch.append(flat.astype(np.int32))
                arrays[f"{view}_tokens"] = torch.cat(token_rows).to(
                    torch.float16).numpy()
                arrays[f"{view}_row"] = np.concatenate(token_row_id)
                arrays[f"{view}_patch"] = np.concatenate(token_patch)
                for scale in SCALES:
                    arrays[f"{view}_{scale}_global"] = l2norm(
                        pooled[(view, scale)]).numpy().astype(np.float32)
                    arrays[f"{view}_{scale}_masked"] = masked[
                        (view, scale, "entity")].numpy().astype(np.float32)
                    arrays[f"{view}_{scale}_background"] = masked[
                        (view, scale, "background")].numpy().astype(np.float32)
                arrays[f"{view}_exclusive"] = masked[
                    (view, "context", "exclusive")].numpy().astype(np.float32)
                arrays[f"{view}_exclusive_count"] = masked[(view, "exclusive_count")]

            # Correspondence in flat token-row space, so the trainer never has to
            # reconstruct which stored row a patch index refers to.
            td_off = np.concatenate([[0], np.cumsum(
                [int(np.sum(p["view"].mask_of("td", "context", PATCH_GRID)))
                 for p in payload])]).astype(np.int64)
            o_off = np.concatenate([[0], np.cumsum(
                [int(np.sum(p["view"].mask_of("oblique", "context", PATCH_GRID)))
                 for p in payload])]).astype(np.int64)
            corr_cand, corr_td, corr_o, corr_w = [], [], [], []
            for i, p in enumerate(payload):
                corr = p["view"].correspondence
                if corr["weight"].size == 0:
                    continue
                td_map = {int(v): j for j, v in enumerate(
                    np.flatnonzero(p["view"].mask_of("td", "context", PATCH_GRID)
                                   .reshape(-1)).tolist())}
                o_map = {int(v): j for j, v in enumerate(
                    np.flatnonzero(p["view"].mask_of("oblique", "context", PATCH_GRID)
                                   .reshape(-1)).tolist())}
                td_local = np.array([td_map[int(v)] for v in corr["td"]], np.int64)
                o_local = np.array([o_map[int(v)] for v in corr["o"]], np.int64)
                corr_cand.append(np.full(len(td_local), i, dtype=np.int32))
                corr_td.append((td_local + td_off[i]).astype(np.int32))
                corr_o.append((o_local + o_off[i]).astype(np.int32))
                corr_w.append(corr["weight"].astype(np.float32))
            arrays["corr_cand"] = (np.concatenate(corr_cand) if corr_cand
                                   else np.zeros(0, np.int32))
            arrays["corr_td_row"] = (np.concatenate(corr_td) if corr_td
                                     else np.zeros(0, np.int32))
            arrays["corr_o_row"] = (np.concatenate(corr_o) if corr_o
                                    else np.zeros(0, np.int32))
            arrays["corr_w"] = (np.concatenate(corr_w) if corr_w
                                else np.zeros(0, np.float32))

            arrays["entity_ids"] = np.array([p["spec"].entity_id for p in payload],
                                            np.int64)
            arrays["entity_types"] = np.array([p["spec"].object_type
                                               for p in payload])
            arrays["is_target"] = np.array(
                [p["spec"].entity_id == sample["target_id"] for p in payload], bool)
            arrays["geometry"] = np.stack([p["view"].geometry for p in payload])
            arrays["visible_point_ratio"] = np.array(
                [p["view"].visible_point_ratio for p in payload], np.float32)
            arrays["occlusion_ratio"] = np.array(
                [p["view"].occlusion_ratio for p in payload], np.float32)
            arrays["support_points"] = np.array(
                [len(p["support"]) for p in payload], np.int64)
            arrays["extent_m"] = np.stack([
                np.array([entity_scales(p["spec"].dimension)[s] for s in SCALES],
                         np.float32) for p in payload])
            arrays["patch_counts"] = np.stack([
                np.array([p["view"].count_of(v, s)
                          for v in ("td", "oblique") for s in SCALES], np.int32)
                for p in payload])
            arrays["support_source"] = np.array(
                [p["support"].source for p in payload])
            for variant in TEXT_VARIANTS:
                text = sample["texts"].get(variant)
                arrays[f"text_{variant}"] = (
                    text_feats[text_index[text]].detach().cpu().numpy().astype(
                        np.float32) if text else np.zeros(0, np.float32))

            stem = f"{split}_{sample['episode_index']:05d}_step{sample['step']:03d}"
            np.savez(out_dir / f"{stem}.npz", **arrays)

            tgt = payload[target_rows[0]]
            record = {
                "split": split, "map": map_name, "kept": True,
                "episode_index": int(sample["episode_index"]),
                "step": int(sample["step"]), "file": f"{stem}.npz",
                "target_id": int(sample["target_id"]),
                "target_type": sample["target_type"],
                "target_distance": float(sample["target_distance"]),
                "target_centre_offset": float(sample["target_centre_offset"]),
                "target_support_source": tgt["support"].source,
                "target_support_points": int(len(tgt["support"])),
                "target_correspondence_pairs": int(
                    tgt["view"].correspondence["weight"].size),
                "target_visible_point_ratio": float(tgt["view"].visible_point_ratio),
                "same_class_candidates": int(sample["same_class_candidates"]),
                "instruction": sample["instruction"],
                "texts": {v: sample["texts"][v] for v in TEXT_VARIANTS
                          if sample["texts"].get(v)},
                "entity_ids": [int(p["spec"].entity_id) for p in payload],
                "entity_types": [p["spec"].object_type for p in payload],
                "candidate_types": sample["candidate_types"],
                "candidate_count": len(payload),
                "oblique_valid_pixel_ratio": float(render.valid_ratio),
                "oblique_in_tile_valid_ratio": render.stats.get("in_tile_valid_ratio"),
                "render_seconds": round(render_s, 3),
                "encode_seconds": round(encode_s, 3),
                "skipped": dict(skipped),
            }
            with shard_path.open("a") as handle:
                handle.write(json.dumps(record, default=float) + "\n")
            stats["kept"] += 1
            type_counts[sample["target_type"]] += 1
            for p in payload:
                stats[f"source::{p['support'].source}"] += 1
            if (n + 1) % 10 == 0:
                print(f"  {split} {n + 1}/{len(samples)} kept={stats['kept']} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        print(f"{split} shard {args.shard}: {dict(stats)}", flush=True)
        print(f"{split} target types: {dict(type_counts)}", flush=True)
        (out_dir / f"{split}.shard{args.shard}.summary.json").write_text(
            json.dumps({"split": split, "shard": args.shard, "shards": args.shards,
                        "stats": dict(stats), "target_types": dict(type_counts)},
                       indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
