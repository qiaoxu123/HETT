#!/usr/bin/env python3
"""Resolution-controlled viewpoint test.

The question this answers: is the oblique45 advantage a viewpoint effect, or a
resolution effect in disguise?

The confound is real and larger than it first looked.  For a landmark at
distance ``d`` the perspective ground sample distance is ``d / focal``, so at
512 px (focal 256) and the probe's median distance of 159 m it is **0.62 m/px**
against 0.1 m/px for the shipped raster -- six times coarser linearly, thirty-six
times in area.  Comparing those two directly measures resolution as much as
viewpoint.

So the same 158 paired samples from Gate B are re-measured with two independent
sweeps, holding everything else fixed:

* the oblique view at 512 / 1024 / 1536 / 2048 px, changing only image sampling
  (camera pose, FOV, near/far and landmark geometry are untouched);
* the top-down crop degraded to 0.2 and 0.3 m/px, and -- per sample -- to the
  exact ground sample distance the oblique view has at that landmark's range.

Every source uses the same physical crop extent for a given landmark, so the
fraction of the crop the landmark occupies is equal by construction and no view
gets a framing advantage.
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

OBLIQUE_PITCH = -45.0
OBLIQUE_SIZES = (512, 1024, 1536, 2048)
TD_FIXED_RESOLUTIONS = (0.2, 0.3)
NATIVE_TD_RESOLUTION = 0.1
MIN_CROP_EXTENT_M = 24.0
MAX_CROP_EXTENT_M = 150.0


def crop_extent_m(dimension) -> float:
    """Physical side of the square crop, identical for every source.

    Using one extent for all sources makes the landmark's *ideal* fraction of
    the crop the same everywhere, so no view is handed a framing advantage; what
    differs is only how much real detail survives at that extent.
    """
    return float(np.clip(2.5 * max(dimension[0], dimension[1]),
                         MIN_CROP_EXTENT_M, MAX_CROP_EXTENT_M))


def crop_square(image, centre_u, centre_v, side_px):
    h, w = image.shape[:2]
    side = int(np.clip(side_px, 16, min(h, w)))
    half = side // 2
    cu, cv_ = int(round(centre_u)), int(round(centre_v))
    c0, r0 = cu - half, cv_ - half
    c1, r1 = c0 + side, r0 + side
    pad_l, pad_t = max(0, -c0), max(0, -r0)
    pad_r, pad_b = max(0, c1 - w), max(0, r1 - h)
    canvas = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    return canvas[r0 + pad_t:r1 + pad_t, c0 + pad_l:c1 + pad_l]


def degrade(image, target_resolution: float, native_resolution: float):
    """Reduce real detail to ``target_resolution`` while keeping the tensor shape.

    Downsample, then resize back up: the encoder receives an image of the same
    size showing less.  Degrading by cropping or by changing the extent instead
    would confound resolution with framing, which is the thing being controlled.
    """
    if target_resolution <= native_resolution * 1.001:
        return image
    h, w = image.shape[:2]
    factor = target_resolution / native_resolution
    small = cv2.resize(image, (max(int(round(w / factor)), 8),
                               max(int(round(h / factor)), 8)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def landmark_pixel_ratio(render_result, camera, landmark, grid, sorted_xyz,
                         target_samples: int = 4096) -> float:
    """Fraction of the whole frame whose winning point lies inside the landmark's box.

    Measured from the render rather than assumed, and subsampled so it stays
    cheap at 2048 px.  This is the target's share of the *frame*; its share of a
    candidate *crop* is ``landmark_pixel_ratio_ideal``, which is equal across
    sources by construction because every source uses the same crop extent.
    """
    _, _, _, pos, dim = landmark
    pos = np.asarray(pos, dtype=np.float64)
    half = np.asarray(dim, dtype=np.float64) / 2.0
    h, w = render_result.point_id.shape
    stride = max(int(np.sqrt(h * w / max(target_samples, 1))), 1)
    slots = render_result.point_id[::stride, ::stride]
    valid = slots >= 0
    if not valid.any():
        return 0.0
    # `point_id` from render_cloud_region is already a bucket-slot, so the
    # position comes straight out of the ordered array.  A few thousand samples
    # are plenty for a coverage ratio, and gathering them individually avoids
    # reading the whole span: a rendered frame can address most of an
    # 80M-point block, and pulling ~1 GB in to sample 4096 pixels was costing
    # more than the render it was measuring.
    world = np.asarray(sorted_xyz[np.asarray(slots[valid], dtype=np.int64)])
    inside = np.all(np.abs(world - pos[None, :]) <= (half + 1.0)[None, :], axis=1)
    return float(inside.mean())


def build_sources(sample, distance):
    """Every (name, kind, parameter) this sample is measured under."""
    sources = [("td_native", "topdown", NATIVE_TD_RESOLUTION)]
    sources += [(f"td_{int(r * 100):03d}", "topdown", r) for r in TD_FIXED_RESOLUTIONS]
    # Per-sample matching: the ground sample distance the oblique view has at
    # this landmark's range, which is what makes the comparison like for like.
    for size in (512, 1536, 2048):
        sources.append((f"td_match_{size}", "topdown", distance / (size / 2.0)))
    sources += [(f"o{size}", "oblique", size) for size in OBLIQUE_SIZES]
    return sources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--gate-b", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    gate_b_dir = Path(args.gate_b) if args.gate_b else artifact_dir(cfg) / "gate_b"
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "resolution_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"

    transform_path = Path(args.transform) if args.transform else \
        artifact_dir(cfg) / "transform" / "best_transform.json"
    transform = load_transform(transform_path) if transform_path.exists() \
        else CoordinateTransform.identity()

    # The Gate B sample set is reused verbatim: same instructions, same
    # candidate ids, same target.  Re-sampling would make this a different
    # experiment rather than a control on the previous one.
    gate_b = json.loads((gate_b_dir / "gate_b.json").read_text())
    rows = [r for r in gate_b["rows"]
            if r["views"].get("topdown") and r["views"].get("oblique45")
            and "phrase" in r["views"]["topdown"].get("sims", {})
            and "phrase" in r["views"]["oblique45"].get("sims", {})]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} paired samples reused from Gate B", flush=True)

    objects_by_map = load_landmarks(cfg)
    splits = {s: citynav.load_split(Path(cfg["paths"]["citynav_dir"]), s)
              for s in {r["split"] for r in rows}}
    episode_of = {(s, e.index): e for s, eps in splits.items() for e in eps}

    from run_gate_b import load_encoder, encode_images, encode_texts, topdown_raster

    processor, model, torch = load_encoder(cfg, args.device)
    print(f"encoder {cfg['vision_probe']['model']} on {args.device}", flush=True)

    unique_texts = sorted({r["texts"][v] for r in rows for v in ("phrase", "name")
                           if r["texts"].get(v)})
    text_feats = encode_texts(processor, model, torch, unique_texts, args.device)
    text_index = {t: i for i, t in enumerate(unique_texts)}

    # Sharding writes each worker to its own file so concurrent appends cannot
    # interleave; they are concatenated afterwards.  Every worker skips what any
    # file already holds, so the union is exactly the sample set.
    base_path = out_dir / "resolution_control.jsonl"
    jsonl_path = base_path if args.shards == 1 else \
        out_dir / f"resolution_control.shard{args.shard}.jsonl"
    done = set()
    for path in sorted(out_dir.glob("resolution_control*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["split"], rec["episode_index"], rec["step"]))
    print(f"resuming with {len(done)} samples already on disk "
          f"(shard {args.shard}/{args.shards})", flush=True)
    cache, rasters = {}, {}
    t0 = time.time()
    for n, row in enumerate(rows):
        if (row["split"], row["episode_index"], row["step"]) in done:
            continue
        if args.shards > 1 and (n % args.shards) != args.shard:
            continue
        map_name = row["map"]
        if map_name not in cache:
            cache = {map_name: build_map_context(cfg, map_name,
                                                 objects_by_map=objects_by_map)}
        ctx = cache[map_name]
        if map_name not in rasters:
            rasters[map_name] = topdown_raster(Path(cfg["paths"]["ortho_dir"]), map_name)
        raster_info = rasters[map_name]

        episode = episode_of[(row["split"], row["episode_index"])]
        position = transform.apply_xyz(
            np.asarray(episode.trajectory[row["step"], :3])[None, :])[0]
        yaw = float(episode.yaw()[row["step"]])
        # Row 0 of the Gate B record is the referenced landmark's own geometry.
        objs = objects_by_map[map_name]
        target = objs[row["target_id"]]

        source_specs = build_sources(row, row["target_distance"])
        rendered = {}
        render_seconds = {}
        for name, kind, param in source_specs:
            if kind != "oblique":
                continue
            size = int(param)
            cam = Camera(position=position, yaw=yaw, pitch=np.deg2rad(OBLIQUE_PITCH),
                         width=size, height=size, hfov_deg=cfg["render"]["hfov_deg"],
                         near=cfg["render"]["near"], far=cfg["render"]["far"])
            t = time.time()
            res = render_cloud_region(ctx.cloud, ctx.grid, cam, splat_radius=0,
                                      lod=cfg["render"]["lod"])
            render_seconds[f"o{size}"] = round(time.time() - t, 3)
            rendered[f"o{size}"] = {"camera": cam, "result": res}

        focal512 = rendered["o512"]["camera"].focal
        sources = {}
        cand_objs = [objs[c] for c in row["candidate_ids"]]
        for cid, cobj in zip(row["candidate_ids"], cand_objs):
            pos = np.asarray(cobj.position, dtype=np.float64)
            dim = [float(v) for v in cobj.dimension]
            extent = crop_extent_m(dim)
            entries = {}
            for name, kind, param in source_specs:
                if kind == "topdown":
                    if raster_info is None:
                        entries[name] = None
                        continue
                    raster, origin, res_native = raster_info
                    col = (pos[0] - origin[0]) / res_native
                    rowp = (origin[1] - pos[1]) / res_native
                    if not (0 <= rowp < raster.shape[0] and 0 <= col < raster.shape[1]):
                        entries[name] = None
                        continue
                    img = crop_square(raster, col, rowp, extent / res_native)
                    img = degrade(img, float(param), res_native)
                    entries[name] = img
                else:
                    size = int(param)
                    cam = rendered[f"o{size}"]["camera"]
                    proj = project_centre(np.asarray(pos, dtype=np.float64), cam)
                    if proj is None or not proj["inside"]:
                        entries[name] = None
                        continue
                    side_px = cam.focal * extent / max(proj["metric"], 1e-3)
                    entries[name] = crop_square(rendered[f"o{size}"]["result"].rgb,
                                                proj["u"], proj["v"], side_px)
            sources[cid] = entries

        target_lm = (target.id, target.name, target.object_type,
                     tuple(float(v) for v in target.position),
                     tuple(float(v) for v in target.dimension))
        sorted_xyz, _ = ctx.grid.sorted_arrays()
        measured_ratio = {
            name: landmark_pixel_ratio(rendered[name]["result"],
                                       rendered[name]["camera"], target_lm,
                                       ctx.grid, sorted_xyz)
            for name in rendered
        }

        entry = {
            "split": row["split"], "map": map_name,
            "episode_index": row["episode_index"], "step": row["step"],
            "target_id": row["target_id"], "target_type": row["target_type"],
            "target_distance": row["target_distance"],
            "same_class_candidates": row["same_class_candidates"],
            "texts": row["texts"],
            "candidate_ids": row["candidate_ids"],
            "candidate_types": row["candidate_types"],
            "crop_extent_m": {c.id: crop_extent_m(o.dimension)
                              for c, o in zip(cand_objs, cand_objs)},
            "landmark_pixel_ratio_ideal": {
                o.id: float((float(o.dimension[0]) * float(o.dimension[1]))
                            / crop_extent_m(o.dimension) ** 2)
                for o in cand_objs},
            "target_pixel_ratio_in_frame": measured_ratio,
            "gaussian_sample_distance_m": {
                name: row["target_distance"] / (int(name[1:]) / 2.0)
                for name in rendered},
            "topdown_resolution_for_source": {
                name: (float(param) if kind == "topdown" else None)
                for name, kind, param in source_specs},
            "candidate_count": len(row["candidate_ids"]),
            "render_seconds": render_seconds,
            "views": {},
        }

        for name, kind, _ in source_specs:
            ids = [c for c in row["candidate_ids"] if sources[c].get(name) is not None]
            if row["target_id"] not in ids or len(ids) < 2:
                continue
            images = []
            for cid in ids:
                img = sources[cid][name]
                if cv2 is not None:
                    d = crops_dir / (f"{row['split']}_{row['episode_index']:05d}"
                                     f"_step{row['step']:03d}")
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{name}_cand{cid}.png"),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                images.append(img)
            feats = encode_images(processor, model, torch, images, args.device)
            sims = {}
            for variant in ("phrase", "name"):
                text = row["texts"].get(variant)
                if not text:
                    continue
                tf = text_feats[text_index[text]:text_index[text] + 1]
                sims[variant] = [float(x) for x in
                                 (feats @ tf.T).squeeze(-1).cpu().numpy()]
            entry["views"][name] = {"ids": [int(c) for c in ids], "sims": sims}

        with jsonl_path.open("a") as handle:
            handle.write(json.dumps(entry, default=float) + "\n")
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)

    lines = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l]
    medians = {k: float(np.median([x["render_seconds"][k] for x in lines]))
               for k in ("o512", "o1024", "o1536", "o2048")}
    print(f"\n{len(lines)} samples -> {jsonl_path}")
    print(f"total {time.time() - t0:.0f}s   median render seconds: "
          + json.dumps(medians))
    print("median top-down GSD matched per sample: "
          + json.dumps({f"{k}": float(np.median(
              [x["topdown_resolution_for_source"][f"td_match_{k}"] for x in lines]))
              for k in (512, 1536, 2048)}))


if __name__ == "__main__":
    main()
