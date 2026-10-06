#!/usr/bin/env python3
"""Step 14: a small SigLIP probe -- does the render carry usable semantic signal?

Not a model experiment.  For poses with a visible referenced landmark this asks
whether a frozen image encoder ranks the referenced landmark's own name above
the other CityRefer candidates in the same map, once for a top-down crop and
once for the rendered views.  Top-1 / Top-4 / margin only.
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

from sensaturban_fpv import citynav  # noqa: E402
from sensaturban_fpv.config import (  # noqa: E402
    artifact_dir, build_map_context, load_config, load_landmarks,
)
from sensaturban_fpv.fit_coordinate_transform import (  # noqa: E402
    CoordinateTransform, load_transform,
)
from sensaturban_fpv.pointcloud_renderer import Camera, render_cloud_region  # noqa: E402
from sensaturban_fpv.project_landmarks import landmark_visibility  # noqa: E402
from sensaturban_fpv.render_trajectory_fpv import (  # noqa: E402
    PoseSample, topdown_context,
)


def _find_spiece(cache_dir: str | None, model_name: str):
    """Locate the SentencePiece model inside an HF cache snapshot.

    SigLIP's processor reads ``vocab_file`` from ``tokenizer_config.json``, and
    the cached SigLIP2 snapshot does not carry that key, so the tokenizer comes
    up with ``vocab_file=None`` and fails on the first encode.  Pointing it at
    the cached ``tokenizer.model`` is enough.
    """
    if not cache_dir:
        return None
    root = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    for candidate in sorted(root.glob("snapshots/*/tokenizer.model")):
        return str(candidate)
    for candidate in sorted(root.glob("snapshots/*/spiece.model")):
        return str(candidate)
    return None


def load_model(model_name: str, cache_dir: str | None, device: str,
               local_files_only: bool = True):
    """Load a frozen encoder; the weights are expected to be already cached."""
    import torch
    from transformers import AutoModel, AutoProcessor

    kwargs = {"local_files_only": local_files_only}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    vocab = _find_spiece(cache_dir, model_name)
    if vocab:
        kwargs["vocab_file"] = vocab
    processor = AutoProcessor.from_pretrained(model_name, **kwargs)
    model = AutoModel.from_pretrained(
        model_name, **{k: v for k, v in kwargs.items() if k != "vocab_file"}
    ).to(device).eval()
    return processor, model, torch


def encode_images(processor, model, torch, images, device, batch=16):
    feats = []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        inputs = processor(images=chunk, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        feats.append(out / out.norm(dim=-1, keepdim=True))
    return torch.cat(feats, dim=0)


def encode_texts(processor, model, torch, texts, device, batch=32, max_length=64):
    """SigLIP's text tower expects fixed-length, padded input.

    ``max_length`` has to be given explicitly: the slow SigLIP tokenizer accepts
    ``padding="max_length"`` but then leaves the sequences ragged, and batching
    fails while building the tensor.
    """
    feats = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        inputs = processor(text=chunk, padding="max_length", truncation=True,
                           max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        feats.append(out / out.norm(dim=-1, keepdim=True))
    return torch.cat(feats, dim=0)


def rank_landmarks(image_feat, text_feats, names, target_index) -> dict:
    """Simple dot-product ranking of landmark names against one image."""
    sims = (image_feat @ text_feats.T).squeeze(0)
    order = np.argsort(-sims.detach().cpu().numpy())
    rank = int(np.where(order == target_index)[0][0]) + 1
    return {
        "rank": rank,
        "top1": bool(rank == 1),
        "top4": bool(rank <= 4),
        "best_sim": float(sims[order[0]]),
        "target_sim": float(sims[target_index]),
        "margin_to_best": float(sims[target_index] - sims[order[0]]),
    }


def summarise(rows: list, key: str) -> dict:
    r = [row[key] for row in rows if row.get(key)]
    if not r:
        return {"n": 0}
    ranks = np.array([x["rank"] for x in r])
    return {
        "n": len(r),
        "top1": float(np.mean([x["top1"] for x in r])),
        "top4": float(np.mean([x["top4"] for x in r])),
        "median_rank": float(np.median(ranks)),
        "mean_margin": float(np.mean([x["margin_to_best"] for x in r])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--transform", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--poses", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-per-map", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["render"]["lod"] = tuple(tuple(x) for x in cfg["render"]["lod"])
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform_path = Path(args.transform) if args.transform else \
        artifact_dir(cfg) / "transform" / "best_transform.json"
    transform = load_transform(transform_path) if transform_path.exists() \
        else CoordinateTransform.identity()

    objects_by_map = load_landmarks(cfg)
    episodes_by_split = {
        split: citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split)
        for split in cfg["sampling"]["per_split"]
    }
    poses_wanted = args.poses or cfg["vision_probe"]["poses"]

    processor, model, torch = load_model(
        cfg["vision_probe"]["model"], cfg["vision_probe"].get("cache_dir"), args.device,
        local_files_only=cfg["vision_probe"].get("local_files_only", True))
    print(f"loaded {cfg['vision_probe']['model']} on {args.device}", flush=True)

    # Collect candidate poses that reference a landmark, grouped by map.
    candidates = defaultdict(list)
    for split, episodes in episodes_by_split.items():
        for ep in episodes:
            if ep.object_ids:
                candidates[ep.map_name].append((split, ep))
    for map_name in candidates:
        candidates[map_name] = candidates[map_name][:args.limit_per_map]

    rows = []
    t0 = time.time()
    for map_name, entries in sorted(candidates.items()):
        objects = objects_by_map.get(map_name, {})
        if not objects:
            continue
        names = [f"a photo of a {o.object_type.lower()}" + (f", {o.name}" if o.name else "")
                 for o in objects.values()]
        ids = [o.id for o in objects.values()]
        text_feats = encode_texts(processor, model, torch, names, args.device)

        ctx = build_map_context(cfg, map_name, objects_by_map=objects_by_map)
        for split, ep in entries:
            for target_id in ep.object_ids[:1]:
                if target_id not in ids:
                    continue
                target_index = ids.index(target_id)
                step = len(ep.trajectory) // 2
                yaw = float(ep.yaw()[step])
                pose4 = np.array([*ep.trajectory[step, :3], yaw])
                pos_ply = transform.apply_xyz(ep.trajectory[step:step + 1, :3])[0]
                sample = PoseSample(split, ep.index, map_name, step, pose4, pos_ply)

                target_lm = next(lm for lm in ctx.landmarks if lm[0] == target_id)
                views = {}
                for name, pitch in cfg["render"]["views"].items():
                    cam = Camera(position=sample.position_ply, yaw=yaw,
                                 pitch=np.deg2rad(pitch),
                                 width=cfg["render"]["width"],
                                 height=cfg["render"]["height"],
                                 hfov_deg=cfg["render"]["hfov_deg"],
                                 near=cfg["render"]["near"], far=cfg["render"]["far"])
                    res = render_cloud_region(ctx.cloud, ctx.grid, cam,
                                              splat_radius=cfg["render"]["splat_radius"],
                                              lod=cfg["render"]["lod"])
                    vis = landmark_visibility(target_lm, cam, res, ctx.cloud, ctx.grid)
                    views[name] = {"rgb": res.rgb, "visible": bool(vis.get("visible")),
                                   "in_fov": bool(vis["in_fov"]),
                                   "valid_ratio": float(res.valid_ratio)}

                td = topdown_context(ctx.cloud, ctx.grid, sample, extent=60.0, resolution=1.0)
                entry = {
                    "map": map_name, "split": split, "episode_index": int(ep.index),
                    "target_id": int(target_id),
                    "target_name": target_lm[1] or target_lm[2],
                    "instruction": ep.description,
                    "num_candidates": len(ids),
                }

                # Top-down crop as the baseline view.
                if td.get("image") is not None:
                    td_feat = encode_images(processor, model, torch, [td["image"]], args.device)
                    entry["topdown"] = rank_landmarks(td_feat[0], text_feats, names, target_index)

                for name, v in views.items():
                    if not v["visible"]:
                        continue
                    feat = encode_images(processor, model, torch, [v["rgb"]], args.device)
                    entry[name] = rank_landmarks(feat[0], text_feats, names, target_index)
                    entry[f"{name}_valid_ratio"] = v["valid_ratio"]

                rows.append(entry)
                print(f"  {map_name} ep{ep.index} target={entry['target_name']!r} "
                      f"visible_views={[k for k, v in views.items() if v['visible']]}",
                      flush=True)
        print(f"{map_name}: {len(entries)} poses ({time.time() - t0:.0f}s elapsed)", flush=True)
        if len(rows) >= poses_wanted:
            break

    summary = {
        "model": cfg["vision_probe"]["model"],
        "poses": len(rows),
        "num_candidates_median": float(np.median([r["num_candidates"] for r in rows]))
        if rows else None,
        "results": {
            key: summarise(rows, key)
            for key in ("topdown", "fpv", "fpv_down10", "oblique", "steep_oblique")
        },
    }
    (out_dir / "vision_probe.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True) + "\n")
    print("\n" + json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {out_dir / 'vision_probe.json'}")


if __name__ == "__main__":
    main()
