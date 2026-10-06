#!/usr/bin/env python3
"""Gate B: does a perspective view carry more landmark identity than top-down?

Same text, same candidate set, same frozen encoder, only the view changes.
Landmark geometry comes from CityRefer and is used to place the crops, because
what is under test is *observability* -- whether the viewpoint shows more of what
identifies the landmark -- not a detector.

Two texts are probed on the identical samples:

``name``      the landmark's own name ("Merton Hall")
``phrase``    CityRefer's referenced phrase ("white triangle shaped building")

A third, the full instruction, is not used: the referenced span is available, so
truncating a whole instruction into a 64-token context would only add noise.

Candidate sets differ between views -- top-down sees every landmark in the block,
a 90 degree oblique sees a handful -- so raw similarity scores are stored per
candidate and every metric is computed afterwards on the *intersection* for the
pair being compared.  A comparison across different candidate sets would not be
a comparison.
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
from sensaturban_fpv.pointcloud_renderer import render_cloud_region  # noqa: E402
from sensaturban_fpv.pose_selection import camera_for  # noqa: E402
from sensaturban_fpv.project_landmarks import project_centre  # noqa: E402

VIEWS = {"fpv": -10.0, "oblique30": -30.0, "oblique45": -45.0}
ALL_VIEWS = ("topdown", "fpv", "oblique30", "oblique45")
PERSPECTIVE = ("fpv", "oblique30", "oblique45")
VARIANTS = ("name", "phrase")


# --------------------------------------------------------------------------
# sample construction
# --------------------------------------------------------------------------

def referenced_texts(obj) -> dict:
    """The two probe texts for one CityRefer object, or ``None`` where absent."""
    name = (obj.name or "").strip()
    phrase = ""
    for pd in getattr(obj, "processed_descriptions", []) or []:
        candidate = (getattr(pd, "target", "") or "").strip()
        if candidate and candidate.lower() not in ("", "building", "object"):
            phrase = candidate
            break
    return {"name": name or None, "phrase": phrase or None}


def build_samples(cfg, objects_by_map, splits, target, min_candidates=4,
                  max_candidates=10, same_class_quota=0.6, max_per_map=40):
    """Referenced-landmark samples with same-scene, same-class-heavy candidates."""
    episodes_by_split = {
        split: citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split)
        for split in splits
    }
    # Squared distances in plain Python floats.  Sorting the candidate list
    # with a key that calls np.asarray and np.linalg.norm per comparison costs
    # minutes per split on train_seen's 22k episodes, times every worker.
    xyz_of = {
        map_name: {oid: tuple(float(v) for v in o.position)
                   for oid, o in objects.items()}
        for map_name, objects in objects_by_map.items()
    }

    def dist2(map_name, a_id, b_id):
        ax, ay, az = xyz_of[map_name][a_id]
        bx, by, bz = xyz_of[map_name][b_id]
        return (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2

    samples, per_map = [], defaultdict(int)
    for split, episodes in episodes_by_split.items():
        for episode in episodes:
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

            same_class = [o for o in others if o.object_type == ref.object_type]
            other_class = [o for o in others if o.object_type != ref.object_type]
            n_same = int(round(max_candidates * same_class_quota))
            # Same-class distractors first: a candidate set of obviously
            # different objects would make the task easy for the wrong reason.
            same_class.sort(key=lambda o: dist2(episode.map_name, o.id, ref.id))
            other_class.sort(key=lambda o: dist2(episode.map_name, o.id, ref.id))
            candidates = ([ref] + same_class[:n_same] + other_class)[:max_candidates]
            if len(candidates) < min_candidates:
                continue

            position = episode.trajectory[:, :3]
            yaws = episode.yaw()
            found = None
            for step in np.linspace(0, len(position) - 1, num=6).astype(int):
                pos = position[step]
                yaw = float(yaws[step])
                distance = float(np.linalg.norm(
                    np.asarray(ref.position, dtype=np.float64) - pos))
                if not (20.0 <= distance <= 220.0):
                    continue
                in_view = {}
                for name, pitch in VIEWS.items():
                    cam = camera_for(pos, yaw, pitch, cfg)
                    proj = project_centre(np.asarray(ref.position, float), cam)
                    in_view[name] = bool(proj and proj["inside"])
                if any(in_view.values()):
                    found = (int(step), pos, yaw, distance, in_view)
                    break
            if found is None or per_map[episode.map_name] >= max_per_map:
                continue
            step, pos, yaw, distance, in_view = found
            per_map[episode.map_name] += 1

            samples.append({
                "split": split, "map": episode.map_name,
                "episode_index": int(episode.index), "step": step,
                "position": pos.tolist(), "yaw_deg": float(np.rad2deg(yaw)),
                "instruction": episode.description,
                "texts": texts,
                "target_id": int(ref.id),
                "target_name": ref.name, "target_type": ref.object_type,
                "target_distance": distance,
                "target_in_view": in_view,
                "candidate_ids": [int(o.id) for o in candidates],
                "candidate_types": [o.object_type for o in candidates],
                "candidate_positions": [[float(v) for v in o.position]
                                        for o in candidates],
                "candidate_dimensions": [[float(v) for v in o.dimension]
                                         for o in candidates],
                "same_class_candidates": int(sum(
                    1 for o in candidates[1:] if o.object_type == ref.object_type)),
            })
            if len(samples) >= target:
                return samples
    return samples


# --------------------------------------------------------------------------
# crops
# --------------------------------------------------------------------------

def crop_square(image, centre_u, centre_v, side_px):
    """Crop centred on a pixel, clamped to the image, padded if it runs off."""
    h, w = image.shape[:2]
    side = int(np.clip(side_px, 16, min(h, w)))
    half = side // 2
    cu, cv_ = int(round(centre_u)), int(round(centre_v))
    c0, r0 = cu - half, cv_ - half
    c1, r1 = c0 + side, r0 + side

    pad_l, pad_t = max(0, -c0), max(0, -r0)
    pad_r, pad_b = max(0, c1 - w), max(0, r1 - h)
    canvas = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r,
                                cv2.BORDER_REPLICATE)
    return canvas[r0 + pad_t:r1 + pad_t, c0 + pad_l:c1 + pad_l]


def topdown_raster(ortho_dir: Path, map_name: str):
    from PIL import Image
    png = Path(ortho_dir) / f"{map_name}.png"
    if not png.exists():
        return None
    raster = np.asarray(Image.open(png).convert("RGB"))
    origin, res = (0.0, 0.0), 0.1
    try:
        import rasterio
        with rasterio.open(Path(ortho_dir) / f"{map_name}.tif") as ds:
            origin = (ds.transform.c, ds.transform.f)
            res = ds.transform.a
    except Exception:
        return None
    return raster, origin, res


def topdown_crops(raster_info, sample):
    if raster_info is None:
        return {}
    raster, origin, res = raster_info
    crops = {}
    for cid, pos, dim in zip(sample["candidate_ids"], sample["candidate_positions"],
                             sample["candidate_dimensions"]):
        col = (pos[0] - origin[0]) / res
        row = (origin[1] - pos[1]) / res
        if not (0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]):
            crops[cid] = None
            continue
        side_m = float(np.clip(2.5 * max(dim[0], dim[1]), 24.0, 150.0))
        crops[cid] = crop_square(raster, col, row, side_m / res)
    return crops


def perspective_crops(sample, views_render):
    """GT crops sized by the landmark's angular extent, so scale is comparable."""
    focal = views_render["fpv"]["camera"].focal
    crops = defaultdict(dict)
    for cid, pos, dim in zip(sample["candidate_ids"], sample["candidate_positions"],
                             sample["candidate_dimensions"]):
        pos = np.asarray(pos, dtype=np.float64)
        for name in PERSPECTIVE:
            entry = views_render[name]
            proj = project_centre(pos, entry["camera"])
            if proj is None or not proj["inside"]:
                crops[name][cid] = None
                continue
            side_m = float(np.clip(2.5 * max(dim[0], dim[1]), 12.0, 200.0))
            side_px = focal * side_m / max(proj["metric"], 1e-3)
            crops[name][cid] = crop_square(entry["result"].rgb, proj["u"],
                                           proj["v"], side_px)
    return crops


# --------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------

def _find_spiece(cache_dir, model_name):
    if not cache_dir:
        return None
    root = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    for pattern in ("snapshots/*/tokenizer.model", "snapshots/*/spiece.model"):
        found = sorted(root.glob(pattern))
        if found:
            return str(found[0])
    return None


def load_encoder(cfg, device):
    import torch
    from transformers import AutoModel, AutoProcessor

    spec = cfg["vision_probe"]
    kwargs = {"cache_dir": spec["cache_dir"],
              "local_files_only": spec.get("local_files_only", True)}
    vocab = _find_spiece(spec.get("cache_dir"), spec["model"])
    if vocab:
        kwargs["vocab_file"] = vocab
    processor = AutoProcessor.from_pretrained(spec["model"], **kwargs)
    model = AutoModel.from_pretrained(
        spec["model"], cache_dir=spec["cache_dir"],
        local_files_only=spec.get("local_files_only", True)).to(device).eval()
    return processor, model, torch


def encode_images(processor, model, torch, images, device, batch=16):
    feats = []
    for i in range(0, len(images), batch):
        inputs = processor(images=images[i:i + batch], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        feats.append(out / out.norm(dim=-1, keepdim=True))
    return torch.cat(feats, dim=0) if feats else None


def encode_texts(processor, model, torch, texts, device, batch=64, max_length=64):
    """SigLIP needs an explicit ``max_length``: the slow tokenizer accepts
    ``padding='max_length'`` and still leaves the sequences ragged."""
    feats = []
    for i in range(0, len(texts), batch):
        inputs = processor(text=texts[i:i + batch], padding="max_length",
                           truncation=True, max_length=max_length,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        feats.append(out / out.norm(dim=-1, keepdim=True))
    return torch.cat(feats, dim=0)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", type=int, default=160)
    ap.add_argument("--splits", nargs="*", default=["val_unseen", "val_seen"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-per-map", type=int, default=40)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "gate_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"

    transform_path = Path(args.transform) if args.transform else \
        artifact_dir(cfg) / "transform" / "best_transform.json"
    transform = load_transform(transform_path) if transform_path.exists() \
        else CoordinateTransform.identity()

    objects_by_map = load_landmarks(cfg)
    samples = build_samples(cfg, objects_by_map, args.splits, args.target,
                            max_per_map=args.max_per_map)
    print(f"{len(samples)} samples over {len({s['map'] for s in samples})} maps; "
          f"splits {json.dumps({k: sum(1 for s in samples if s['split'] == k) for k in args.splits})}",
          flush=True)
    if not samples:
        print("no samples; nothing to probe")
        return

    processor, model, torch = load_encoder(cfg, args.device)
    print(f"encoder {cfg['vision_probe']['model']} on {args.device}", flush=True)

    unique_texts = sorted({s["texts"][v] for s in samples for v in VARIANTS
                           if s["texts"].get(v)})
    text_feats = encode_texts(processor, model, torch, unique_texts, args.device)
    text_index = {t: i for i, t in enumerate(unique_texts)}
    print(f"{len(unique_texts)} unique texts encoded", flush=True)

    jsonl_path = out_dir / "gate_b.jsonl"
    jsonl_path.write_text("")
    rows, cache = [], {}
    t0 = time.time()
    for n, sample in enumerate(samples):
        ctx = cache.get(sample["map"])
        if ctx is None:
            ctx = build_map_context(cfg, sample["map"], objects_by_map=objects_by_map)
            cache = {sample["map"]: ctx}

        pos_ply = transform.apply_xyz(np.asarray(sample["position"])[None, :])[0]
        views_render = {}
        for name, pitch in VIEWS.items():
            cam = camera_for(pos_ply, np.deg2rad(sample["yaw_deg"]), pitch, cfg)
            res = render_cloud_region(ctx.cloud, ctx.grid, cam,
                                      splat_radius=cfg["render"]["splat_radius"],
                                      lod=cfg["render"]["lod"])
            views_render[name] = {"camera": cam, "result": res}

        crops = {"topdown": topdown_crops(
            topdown_raster(Path(cfg["paths"]["ortho_dir"]), sample["map"]), sample)}
        crops.update(perspective_crops(sample, views_render))

        entry = {k: sample[k] for k in
                 ("split", "map", "episode_index", "step", "instruction", "texts",
                  "target_id", "target_name", "target_type", "target_distance",
                  "target_in_view", "candidate_ids", "candidate_types",
                  "candidate_dimensions", "same_class_candidates")}
        entry["candidate_count"] = len(sample["candidate_ids"])
        entry["views"] = {}

        for view in ALL_VIEWS:
            view_crops = crops.get(view) or {}
            ids = [cid for cid in sample["candidate_ids"]
                   if view_crops.get(cid) is not None]
            if sample["target_id"] not in ids or len(ids) < 2:
                continue
            images = [_save_tile(crops_dir, sample, view, cid, view_crops[cid])
                      for cid in ids]
            feats = encode_images(processor, model, torch, images, args.device)
            sims = {}
            for variant in VARIANTS:
                text = sample["texts"].get(variant)
                if not text:
                    continue
                tf = text_feats[text_index[text]:text_index[text] + 1]
                sims[variant] = [float(x) for x in
                                 (feats @ tf.T).squeeze(-1).cpu().numpy()]
            if sims:
                entry["views"][view] = {"ids": [int(c) for c in ids], "sims": sims}

        rows.append(entry)
        with jsonl_path.open("a") as handle:
            handle.write(json.dumps(entry, default=float) + "\n")
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(samples)} ({time.time() - t0:.0f}s)", flush=True)

    summary = summarise(rows)
    report = {"encoder": cfg["vision_probe"]["model"], "samples": len(rows),
              "summary": summary, "rows": rows}
    (out_dir / "gate_b.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")

    print("\n--- main table ---")
    for variant in VARIANTS:
        for view in ALL_VIEWS:
            s = summary["main_table"].get(variant, {}).get(view)
            if s:
                print(f"  SigLIP2 {variant:6s} {view:10s} n={s['n']:4d} "
                      f"top1={s['top1']:.3f} top4={s['top4']:.3f} "
                      f"mrr={s['mrr']:.3f} margin={s['margin']:+.4f}")
    print("\n--- paired vs top-down ---")
    for variant in VARIANTS:
        for view, t in summary["paired_transitions"].get(variant, {}).items():
            print(f"  {variant:6s} td vs {view:10s} n={t['n']:4d} "
                  f"fail->success={t['td_fail_fpv_success']:3d} "
                  f"success->fail={t['td_success_fpv_fail']:3d} "
                  f"both_ok={t['both_success']:3d} both_bad={t['both_fail']:3d} "
                  f"margin_delta={t['mean_margin_delta_vs_topdown']:+.4f}")
    print(f"\nverdict: {summary['verdict']} "
          f"(best {summary['best']}, "
          f"top1 delta {summary['top1_delta_vs_topdown']})")
    print(f"\nwrote {out_dir / 'gate_b.json'}")


def _save_tile(root: Path, sample, view, cid, image):
    d = root / f"{sample['split']}_{sample['episode_index']:05d}_step{sample['step']:03d}"
    d.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        cv2.imwrite(str(d / f"{view}_cand{cid}.png"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return image


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def rank_within(sims: np.ndarray, target_index: int) -> dict:
    order = np.argsort(-sims)
    rank_pos = int(np.where(order == target_index)[0][0]) + 1
    others = [sims[i] for i in range(len(sims)) if i != target_index]
    best_other = max(others) if others else float("nan")
    return {"rank": rank_pos, "top1": rank_pos == 1, "top4": rank_pos <= 4,
            "mrr": 1.0 / rank_pos, "margin": float(sims[target_index] - best_other)}


def _intersection_metrics(row, view, variant, target_id):
    """Metrics for a pair of views computed on the candidate set they share."""
    td = row["views"].get("topdown")
    other = row["views"].get(view)
    if not td or not other:
        return None
    if variant not in td.get("sims", {}) or variant not in other.get("sims", {}):
        return None
    common = [cid for cid in td["ids"] if cid in set(other["ids"])]
    if target_id not in common or len(common) < 2:
        return None
    idx = common.index(target_id)
    a = np.array([td["sims"][variant][td["ids"].index(c)] for c in common])
    b = np.array([other["sims"][variant][other["ids"].index(c)] for c in common])
    return {"topdown": rank_within(a, idx), "other": rank_within(b, idx),
            "candidates": len(common)}


def summarise(rows) -> dict:
    main = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(dict)
    same_class = defaultdict(lambda: defaultdict(list))
    distance = defaultdict(lambda: defaultdict(list))

    for row in rows:
        target_id = row["target_id"]
        type_of = dict(zip(row["candidate_ids"], row["candidate_types"]))
        n_same_all = sum(1 for t in row["candidate_types"][1:]
                         if t == row["target_type"])
        for variant in VARIANTS:
            if not row["texts"].get(variant):
                continue
            for view in ALL_VIEWS:
                view_data = row["views"].get(view)
                if not view_data or variant not in view_data.get("sims", {}):
                    continue
                ids = view_data["ids"]
                if target_id not in ids or len(ids) < 2:
                    continue
                sims = np.array(view_data["sims"][variant])
                m = rank_within(sims, ids.index(target_id))
                # Carry the bucketing keys on the metric itself: the buckets are
                # computed from the same per-view entries as the main table, so
                # the two can never disagree about which samples they cover.
                m["same_class"] = sum(1 for cid in ids
                                      if cid != target_id
                                      and type_of.get(cid) == row["target_type"])
                m["distance"] = row["target_distance"]
                main[variant][view].append(m)

            for view in PERSPECTIVE:
                pair = _intersection_metrics(row, view, variant, target_id)
                if pair is None:
                    continue
                t, o = pair["topdown"], pair["other"]
                stats = paired.setdefault(variant, {}).setdefault(
                    view, {"td_fail_fpv_success": 0, "td_success_fpv_fail": 0,
                           "both_success": 0, "both_fail": 0, "margins": []})
                if t["top1"] and o["top1"]:
                    stats["both_success"] += 1
                elif not t["top1"] and o["top1"]:
                    stats["td_fail_fpv_success"] += 1
                elif t["top1"] and not o["top1"]:
                    stats["td_success_fpv_fail"] += 1
                else:
                    stats["both_fail"] += 1
                stats["margins"].append(o["margin"] - t["margin"])

    def agg(entries):
        return {"n": len(entries),
                "top1": float(np.mean([e["top1"] for e in entries])),
                "top4": float(np.mean([e["top4"] for e in entries])),
                "mrr": float(np.mean([e["mrr"] for e in entries])),
                "margin": float(np.mean([e["margin"] for e in entries]))}

    main_table = {v: {view: agg(e) for view, e in views.items()}
                  for v, views in main.items()}

    transitions = {}
    for variant, views in paired.items():
        transitions[variant] = {}
        for view, stats in views.items():
            total = (stats["td_fail_fpv_success"] + stats["td_success_fpv_fail"]
                     + stats["both_success"] + stats["both_fail"])
            transitions[variant][view] = {
                "n": total,
                "td_fail_fpv_success": stats["td_fail_fpv_success"],
                "td_success_fpv_fail": stats["td_success_fpv_fail"],
                "both_success": stats["both_success"],
                "both_fail": stats["both_fail"],
                "mean_margin_delta_vs_topdown": (
                    float(np.mean(stats["margins"])) if stats["margins"] else None),
            }

    def bucket(key_fn):
        out = {}
        for variant, views in main.items():
            out[variant] = {}
            for view, entries in views.items():
                groups = defaultdict(list)
                for e in entries:
                    groups[key_fn(e)].append(e)
                out[variant][view] = {k: agg(v) for k, v in sorted(groups.items())}
        return out

    same_class_buckets = bucket(
        lambda e: "0" if e["same_class"] == 0 else ("1-3" if e["same_class"] <= 3 else ">=4"))
    distance_buckets = {}
    for variant, views in main.items():
        distance_buckets[variant] = {}
        for view, entries in views.items():
            groups = defaultdict(list)
            for e in entries:
                d = e["distance"]
                groups["near<50m" if d < 50 else
                       ("medium50-100m" if d < 100 else "far>100m")].append(e)
            distance_buckets[variant][view] = {k: agg(v) for k, v in sorted(groups.items())}

    best, delta = None, None
    for variant in VARIANTS:
        if variant not in main_table or "topdown" not in main_table[variant]:
            continue
        for view in PERSPECTIVE:
            if view not in main_table[variant]:
                continue
            d = main_table[variant][view]["top1"] - main_table[variant]["topdown"]["top1"]
            if delta is None or d > delta:
                delta, best = d, f"{variant}/{view}"

    return {
        "main_table": main_table,
        "paired_transitions": transitions,
        "same_class_buckets": same_class_buckets,
        "distance_buckets": distance_buckets,
        "best": best,
        "top1_delta_vs_topdown": delta,
        "verdict": _verdict(delta),
    }


def _verdict(delta) -> str:
    if delta is None:
        return "no comparison"
    if delta >= 0.10:
        return "strong pass"
    if delta >= 0.05:
        return "weak pass"
    return "fail"


if __name__ == "__main__":
    main()
