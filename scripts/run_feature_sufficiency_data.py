#!/usr/bin/env python3
"""Structured feature blocks for the samples the entity round already fixed.

The candidate sets are *not* resampled.  Every block here is computed for the
same instruction, the same target, the same candidate ids in the same order, on
the same split, episode and step as `artifacts/entity_grounding/data` -- so a
number from this round is directly comparable with the 0.320 that round's best
single view reached, and no feature can be credited for having been measured on
an easier subset.

Four families are produced per candidate -- appearance, geometry, relation,
semantic -- plus the instruction's own relation flags and text embeddings.  An
ORACLE anchor-relation block is produced alongside and written under a name that
says so.

The top-down raster supplies the image-domain appearance: it is a file already
on disk, so this pass needs no render at all and costs seconds per sample rather
than the ~10 s a render-based pass cost.
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

from sensaturban_fpv import citynav  # noqa: E402
from sensaturban_fpv.config import (  # noqa: E402
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.entity_features import (  # noqa: E402
    CLASS_NAMES, ROAD_CLASS, RoadIndex, anchor_relation_features,
    appearance_features, geometry_features, relation_features, semantic_features,
)
from sensaturban_fpv.entity_geometry import (  # noqa: E402
    CLASS_IDS, entity_bounds, polygon_membership, project_point,
)
from sensaturban_fpv.pointcloud_renderer import Camera  # noqa: E402
from sensaturban_fpv.relation_parser import (  # noqa: E402
    instruction_buckets, parse_attributes, parse_relations, relation_feature_vector,
)
from run_gate_b import encode_texts, load_encoder, topdown_raster  # noqa: E402

OBLIQUE_PITCH = -45.0
OBLIQUE_SIZE = 2048
ROAD_QUERY_RADIUS_M = 45.0
MAX_ROAD_POINTS = 40000
# The target's own annotation is the only per-sample reference CityRefer gives,
# so a *deployable* anchor cannot be read off the annotation.  The oracle arm
# synthesises one and is labelled ORACLE everywhere it appears.
ANCHOR_MAX_M = 120.0


def entity_points_and_colours(ctx, spec, target_points: int = 40000):
    """The entity's own points and colours, in one read of the bucket cache.

    Everything this pass computes about an entity is a statistic -- colour
    histograms, spans, densities -- and a 50 m building supplies most of a
    million points, whose read costs more than the rest of the pass together.
    Cells are taken whole but every n-th of them, with n chosen from the bucket
    counts so the expected sample is ``target_points``; a systematic sample of
    the spatial buckets is unbiased for these statistics in a way that
    truncating a sorted read would not be.
    """
    lo, hi = entity_bounds(spec["position"], spec["dimension"], 1.0)
    starts, ends = ctx.grid.query_rect_cell_ranges(lo[0], lo[1], hi[0], hi[1])
    if starts.size == 0:
        return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.uint8)
    per_cell = ends - starts
    total = int(per_cell.sum())
    stride = int(max(1, np.ceil(total / max(target_points, 1))))
    xyz = ctx.grid.read_cell_ranges(spec["sorted_xyz"], starts, ends,
                                    cell_stride=stride)
    rgb = ctx.grid.read_cell_ranges(spec["cloud_rgb"], starts, ends,
                                    cell_stride=stride)
    inside = np.all((xyz >= lo[None, :]) & (xyz <= hi[None, :]), axis=1)
    if spec["contour"] is not None:
        inside &= polygon_membership(xyz[:, :2], spec["contour"])
    class_id = CLASS_IDS.get(spec["object_type"]) if spec["labels"] is not None else None
    if class_id is not None:
        # The semantic class is what keeps a car's appearance from being the
        # road it is parked on, so it is applied here exactly as the entity
        # round applied it.  It costs one byte per point on the same cells.
        labels = ctx.grid.read_cell_ranges(spec["labels"], starts, ends,
                                           cell_stride=stride)
        if len(labels) == len(xyz):
            inside &= (labels == class_id)
    return xyz[inside], rgb[inside]


def candidate_blocks(ctx, spec, uav_position, uav_yaw, raster_info, camera,
                     others_xyz, others_type, block_bounds, road_stats):
    """Every feature family for one candidate entity."""
    pts, rgb = entity_points_and_colours(ctx, spec)
    if len(pts) == 0:
        return None, None

    # Image-domain appearance: the top-down pixels the entity's own points land
    # on, which is the projection the masks were built from.  The raster is a
    # file already on disk, so this arm needs no render.
    patch = None
    projected_px = 0.0
    if raster_info is not None:
        raster, origin, res = raster_info
        col = (pts[:, 0] - origin[0]) / res
        row = (origin[1] - pts[:, 1]) / res
        inside = ((row >= 0) & (row < raster.shape[0])
                  & (col >= 0) & (col < raster.shape[1]))
        if inside.any():
            rr = np.clip(np.round(row[inside]).astype(np.int64), 0, raster.shape[0] - 1)
            cc = np.clip(np.round(col[inside]).astype(np.int64), 0, raster.shape[1] - 1)
            patch = raster[rr, cc]
            projected_px = float(inside.sum())
            if rgb is None or len(rgb) == 0:
                rgb = patch

    appearance = appearance_features(rgb, patch=patch)
    proj = project_point(np.asarray(spec["position"], float), camera)
    oblique_px = 0.0
    if proj is not None:
        # Angular size: what the entity would occupy in the oblique frame, which
        # is the only part of "how visible is it" the geometry alone can say.
        oblique_px = float(camera.focal * max(spec["dimension"][0], spec["dimension"][1])
                           / max(proj["metric"], 1e-3))
        if not (0 <= proj["u"] < camera.width and 0 <= proj["v"] < camera.height):
            oblique_px = -1.0
    geometry = geometry_features(pts, spec["position"], spec["dimension"],
                                 uav_position, uav_yaw, projected_px=projected_px,
                                 patch_count=oblique_px)
    relation = relation_features(spec["position"], uav_position, uav_yaw,
                                 others_xyz, others_type, spec["object_type"],
                                 road=road_stats,
                                 others_class=spec.get("others_class"))
    semantic = semantic_features(spec["object_type"], spec["position"], block_bounds,
                                 entity_name=spec.get("name", ""),
                                 n_entities_in_map=len(others_type) + 1)
    return {"appearance": appearance, "geometry": geometry, "relation": relation,
            "semantic": semantic,
            "projected_px": np.array([projected_px, oblique_px], np.float32)}, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None,
                    help="the entity-round sample directory to reuse verbatim")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = Path(args.source) if args.source else \
        artifact_dir(cfg) / "entity_grounding" / "data"
    out_dir = Path(args.out) if args.out else \
        artifact_dir(cfg) / "feature_sufficiency" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_by_map = load_landmarks(cfg)
    processor, model, torch = load_encoder(cfg, args.device)

    episodes = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        for ep in citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split):
            episodes[(split, ep.index)] = ep

    records = []
    for split in ("train_seen", "val_seen", "val_unseen"):
        for path in sorted(src.glob(f"{split}*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("kept"):
                        records.append(record)
    seen, unique = set(), []
    for record in sorted(records, key=lambda r: (r["split"], r["episode_index"], r["step"])):
        key = (record["split"], record["episode_index"], record["step"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    todo = [r for i, r in enumerate(unique) if i % args.shards == args.shard]
    print(f"{len(unique)} samples, shard {args.shard} takes {len(todo)}", flush=True)

    out_path = out_dir / f"features.shard{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["split"], r["episode_index"], r["step"]))
        print(f"  resuming with {len(done)} done", flush=True)

    text_cache, ctx_cache, raster_cache, road_cache = {}, {}, {}, {}
    map_entities = {}
    stats = defaultdict(int)
    t0 = time.time()
    for n, record in enumerate(todo):
        key = (record["split"], record["episode_index"], record["step"])
        if key in done:
            continue
        map_name = record["map"]
        if map_name not in ctx_cache:
            ctx_cache = {map_name: build_map_context(cfg, map_name,
                                                     objects_by_map=objects_by_map)}
            raster_cache = {map_name: topdown_raster(Path(cfg["paths"]["ortho_dir"]),
                                                     map_name)}
        ctx = ctx_cache[map_name]
        raster_info = raster_cache[map_name]
        sorted_xyz, sorted_rgb = ctx.grid.sorted_arrays()
        labels = ctx.grid.sorted_labels()
        if map_name not in road_cache:
            road_cache = {map_name: RoadIndex(ctx.grid, labels)}
        road = road_cache[map_name]

        episode = episodes.get((record["split"], record["episode_index"]))
        if episode is None:
            stats["missing_episode"] += 1
            continue
        step = int(record["step"])
        uav_position = np.asarray(episode.trajectory[step, :3], dtype=np.float64)
        uav_yaw = float(episode.yaw()[step])
        camera = Camera(position=uav_position, yaw=uav_yaw,
                        pitch=np.deg2rad(OBLIQUE_PITCH), width=OBLIQUE_SIZE,
                        height=OBLIQUE_SIZE, hfov_deg=cfg["render"]["hfov_deg"],
                        near=cfg["render"]["near"], far=cfg["render"]["far"])

        objs = objects_by_map[map_name]
        entity_ids = [int(i) for i in record["entity_ids"]]
        # Neighbourhood statistics are computed against the *whole map*, not
        # against the candidate list.  The candidate list is built around the
        # reference -- its six nearest same-class entities are in it by
        # construction -- so a "how many same-class entities are near me"
        # feature measured over the candidates says "I am the one the list was
        # built around" and identifies the answer without looking at the scene.
        if map_name not in map_entities:
            all_ids = list(objs.keys())
            _names = list(CLASS_NAMES)
            map_entities = {map_name: (
                all_ids,
                np.array([list(map(float, objs[i].position)) for i in all_ids],
                         dtype=np.float64),
                [objs[i].object_type for i in all_ids],
                np.array([_names.index(objs[i].object_type)
                          if objs[i].object_type in _names else -1
                          for i in all_ids], dtype=np.int64))}
        all_ids, all_xyz, all_type, all_class = map_entities[map_name]
        block_bounds = ctx.grid.exact_bounds()
        target_id = int(record["target_id"])
        target_row = entity_ids.index(target_id) if target_id in entity_ids else -1
        is_target = np.array([i == target_id for i in entity_ids])

        # ORACLE anchor: the nearest other entity of a different type to the
        # target.  It is chosen with the target's own annotation, so it is an
        # upper bound on what knowing the language's anchor would buy, never a
        # method.
        cand_xyz = np.array([list(map(float, objs[i].position)) for i in entity_ids],
                            dtype=np.float64)
        cand_type = [objs[i].object_type for i in entity_ids]
        anchor_row = -1
        if target_row >= 0:
            # ORACLE: the anchor is chosen with the target's own annotation, so
            # it bounds what perfect parsing of "beside the church" would buy.
            tpos = cand_xyz[target_row]
            best = None
            for j, (pos, typ) in enumerate(zip(cand_xyz, cand_type)):
                if j == target_row or typ == cand_type[target_row]:
                    continue
                d = float(np.linalg.norm(pos - tpos))
                if d <= ANCHOR_MAX_M and (best is None or d < best[0]):
                    best = (d, j)
            anchor_row = best[1] if best else -1

        blocks = defaultdict(list)
        kept = []
        for j, eid in enumerate(entity_ids):
            obj = objs[eid]
            spec = {
                "position": obj.position, "dimension": obj.dimension,
                "object_type": obj.object_type, "contour": obj.contour,
                "name": obj.name, "sorted_xyz": sorted_xyz, "labels": labels,
                "cloud_rgb": sorted_rgb,
            }
            keep = np.array([i != eid for i in all_ids], dtype=bool)
            spec["others_class"] = all_class[keep]
            out, _ = candidate_blocks(
                ctx, spec, uav_position, uav_yaw, raster_info, camera,
                all_xyz[keep], [t for t, k in zip(all_type, keep) if k],
                block_bounds, road.distance_m(*spec["position"][:2]))
            if out is None:
                stats["no_support"] += 1
                continue
            kept.append(j)
            for family in ("appearance", "geometry", "relation", "semantic",
                           "projected_px"):
                blocks[family].append(out[family])
            if anchor_row >= 0:
                blocks["anchor_relation"].append(anchor_relation_features(
                    obj.position, cand_xyz[anchor_row], cand_type[anchor_row],
                    obj.object_type, uav_position, uav_yaw))
            else:
                blocks["anchor_relation"].append(
                    np.zeros(11, dtype=np.float32))
        if not kept or target_row not in kept:
            stats["target_dropped"] += 1
            continue

        instruction = record.get("instruction", "")
        texts = {"full": instruction}
        for variant in ("phrase", "name"):
            if record["texts"].get(variant):
                texts[variant] = record["texts"][variant]
        new_texts = sorted({t for k, t in texts.items() if k not in text_cache})
        # Embeddings for every instruction variant, once.
        pending = [t for t in {texts.get(v) for v in ("phrase", "name")}
                   if t and t not in text_cache]
        if texts["full"] not in text_cache:
            pending.append(texts["full"])
        if pending:
            feats = encode_texts(processor, model, torch, pending, args.device)
            for text, feat in zip(pending, feats):
                text_cache[text] = feat.detach().cpu().numpy().astype(np.float32)
        del new_texts

        payload = {
            "split": record["split"], "map": map_name,
            "episode_index": int(record["episode_index"]), "step": step,
            "key": record["file"],
            "entity_ids": [entity_ids[j] for j in kept],
            "entity_types": [cand_type[j] for j in kept],
            "is_target": np.asarray(is_target)[kept].astype(bool).tolist(),
            "anchor_row": (kept.index(anchor_row) if anchor_row in kept else -1),
            "instruction": instruction,
            "buckets": sorted(instruction_buckets(instruction)),
            "relations": parse_relations(instruction),
            "attributes": parse_attributes(instruction),
            "relation_flags": relation_feature_vector(instruction),
            "text_full": text_cache[texts["full"]].tolist(),
            "text_phrase": (text_cache[texts["phrase"]].tolist()
                            if texts.get("phrase") else []),
            "text_name": (text_cache[texts["name"]].tolist()
                          if texts.get("name") else []),
        }
        for family, values in blocks.items():
            payload[family] = np.stack(values).tolist()
        with out_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        stats["kept"] += 1
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(todo)} kept={stats['kept']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"shard {args.shard}: {dict(stats)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
