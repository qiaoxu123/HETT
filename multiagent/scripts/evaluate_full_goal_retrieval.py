#!/usr/bin/env python3
"""Full RGB gate, interpretable abstraction ladder, and controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import binomtest

from multiagent.visual_goal.abstraction import VisualMasks, controlled_ablation, make_levels
from multiagent.visual_goal.encoder import build_encoder
from multiagent.visual_goal.metrics import hard_negative_metrics, partial_distance_correlation, retrieval_metrics
from multiagent.visual_goal.leakage import assert_visual_only_inputs


MODELS = {
    "siglip": "google/siglip-base-patch16-224",
    "siglip2": "google/siglip2-so400m-patch16-512",
    "siglip2_partial": "train_seen partial fine-tune of google/siglip2-so400m-patch16-512",
    "dinov2_small": "facebook/dinov2-small",
}


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _open(path):
    return Image.open(path).convert("RGB")


def _encode_paths(encoder, paths, batch_size=32, transforms=None):
    import torch
    import torch.nn.functional as F
    all_chunks = []
    model, processor, device = encoder.model, encoder.processor, encoder.device
    for start in range(0, len(paths), batch_size):
        imgs = []
        for local_i, path in enumerate(paths[start:start + batch_size]):
            image = _open(path)
            if transforms:
                image = transforms(path, image, start + local_i)
            imgs.append(image)
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        assert_visual_only_inputs(inputs)
        with torch.inference_mode():
            if hasattr(model, "get_image_features"):
                feats = model.get_image_features(**inputs)
                if hasattr(feats, "pooler_output"):
                    feats = feats.pooler_output
            else:
                output = model(**inputs)
                feats = getattr(output, "pooler_output", None)
                if feats is None:
                    feats = output.last_hidden_state[:, 0]
        all_chunks.append(F.normalize(feats.float(), dim=-1).cpu().numpy())
    return np.concatenate(all_chunks, axis=0) if all_chunks else np.zeros((0, 1), dtype=np.float32)


def _candidate_path(root, record):
    return root / record["image_path"]


def _masks(root, record):
    data = np.load(root / record["mask_path"])
    return VisualMasks(*(data[k].astype(bool) for k in
                         ("target", "anchor", "building", "road", "parking", "vegetation", "open_area")))


def _transform_lookup(root, record, image, mode, colors=8):
    rgb = np.asarray(image)
    result = make_levels(rgb, _masks(root, record), colors)
    return Image.fromarray(result[mode])


def _dataset_metrics(qf, cf, queries, candidates):
    metrics = retrieval_metrics(
        qf, cf,
        [r["scene_key"] for r in queries], [r["scene_key"] for r in candidates],
        [r["map_name"] for r in queries], [r["map_name"] for r in candidates],
        query_distances=[r["distance_to_goal_m"] for r in queries],
        query_episode_ids=[r["episode_key"] for r in queries],
    )
    metrics["negative_pools"] = hard_negative_metrics(qf, cf, queries, candidates)
    return metrics


def _gate(metrics):
    hard = metrics["negative_pools"]["same_map"]
    if not hard["n"]:
        return {"pass": False, "reason": "no same-map negatives"}
    # Paired positive-vs-hardest-negative win test on validation-seen only.
    p = float(binomtest(round(hard["accuracy"] * hard["n"]), hard["n"], 0.5,
                        alternative="greater").pvalue)
    distance_rho = metrics.get("distance_spearman_rho")
    distance_p = metrics.get("distance_spearman_pvalue")
    passed = bool(hard["accuracy"] >= 0.55 and hard["margin"] > 0 and p < 0.05
                  and distance_rho is not None and distance_rho < 0 and distance_p is not None and distance_p < 0.05)
    return {"pass": passed, "same_map_accuracy": hard["accuracy"],
            "same_map_margin": hard["margin"], "paired_sign_test_p": p,
            "distance_spearman_rho": distance_rho, "distance_spearman_pvalue": distance_p,
            "criterion": "same-map hard-negative accuracy >= .55, positive margin, paired p < .05, and similarity decreases with distance (Spearman p < .05)",
            "reason": "Full RGB retrieves same-map targets and similarity tracks approach on val_seen" if passed else "full RGB did not clear practical same-map/approach-trend gate"}


def _score_abstraction(root, encoder, split, extent, batch_size):
    candidates = [r for r in _read_jsonl(root / f"templates_{split}.jsonl") if r["extent_m"] == extent]
    queries = _read_jsonl(root / f"queries_{split}.jsonl")
    qf = _encode_paths(encoder, [_candidate_path(root, r) for r in queries], batch_size)
    out = {}
    for level in [f"L{i}" for i in range(10)]:
        def level_transform(path, image, index, mode=level):
            candidate = candidates[index]
            return _transform_lookup(root, candidate, image, mode)
        cf = _encode_paths(encoder, [_candidate_path(root, r) for r in candidates], batch_size,
                           transforms=level_transform)
        out[level] = _dataset_metrics(qf, cf, queries, candidates)
    # Posterization color-count sweep is a controlled sub-level of L4.
    for colors in (4, 8, 16):
        def posterized(path, image, index, n=colors):
            record = candidates[index]
            arr = np.asarray(image)
            from multiagent.visual_goal.abstraction import _posterize
            return Image.fromarray(_posterize(arr, n))
        cf = _encode_paths(encoder, [_candidate_path(root, r) for r in candidates], batch_size,
                           transforms=posterized)
        out[f"L4_{colors}colors"] = _dataset_metrics(qf, cf, queries, candidates)
    return out


def _score_controlled_ablation(root, encoder, split, extent, batch_size):
    candidates = [r for r in _read_jsonl(root / f"templates_{split}.jsonl") if r["extent_m"] == extent]
    queries = _read_jsonl(root / f"queries_{split}.jsonl")
    qf = _encode_paths(encoder, [_candidate_path(root, r) for r in queries], batch_size)
    out = {}
    names = ("color_full", "color_gray", "color_4", "color_8", "color_16", "color_none",
             "texture_full", "texture_blur", "texture_strong_blur", "texture_none",
             "geometry_full", "geometry_contour", "geometry_coarse_footprint", "geometry_none",
             "context_no_road", "context_no_parking", "context_no_buildings",
             "context_no_vegetation", "context_no_open_area",
             "target_only", "anchor_only", "target_anchor", "target_anchor_context")
    for name in names:
        images = []
        for r in candidates:
            rgb = np.asarray(_open(root / r["image_path"]))
            images.append(Image.fromarray(controlled_ablation(rgb, _masks(root, r))[name]))
        cf = _encode_image_list(encoder, images, batch_size)
        out[name] = _dataset_metrics(qf, cf, queries, candidates)
    return out


def _encode_image_list(encoder, images, batch_size):
    import torch
    import torch.nn.functional as F
    chunks = []
    for start in range(0, len(images), batch_size):
        inputs = encoder.processor(images=images[start:start + batch_size], return_tensors="pt").to(encoder.device)
        assert_visual_only_inputs(inputs)
        with torch.inference_mode():
            if hasattr(encoder.model, "get_image_features"):
                feats = encoder.model.get_image_features(**inputs)
                if hasattr(feats, "pooler_output"):
                    feats = feats.pooler_output
            else:
                out = encoder.model(**inputs)
                feats = getattr(out, "pooler_output", None)
                if feats is None:
                    feats = out.last_hidden_state[:, 0]
        chunks.append(F.normalize(feats.float(), dim=-1).cpu().numpy())
    return np.concatenate(chunks)


def _save_visual_grids(root, encoder, split, extent, output_dir, count=100, batch_size=24):
    candidates = [r for r in _read_jsonl(root / f"templates_{split}.jsonl") if r["extent_m"] == extent]
    queries = _read_jsonl(root / f"queries_{split}.jsonl")
    if not queries or not candidates:
        return
    qf = _encode_paths(encoder, [_candidate_path(root, r) for r in queries], batch_size)
    cf = _encode_paths(encoder, [_candidate_path(root, r) for r in candidates], batch_size)
    qn = qf / np.maximum(np.linalg.norm(qf, axis=1, keepdims=True), 1e-8)
    cn = cf / np.maximum(np.linalg.norm(cf, axis=1, keepdims=True), 1e-8)
    sims = qn @ cn.T
    rng = np.random.default_rng(11)
    inds = np.linspace(0, len(queries) - 1, min(count, len(queries)), dtype=int)
    panel_w, panel_h = 180, 260
    out = Image.new("RGB", (panel_w * 11, panel_h * len(inds)), "white")
    draw = ImageDraw.Draw(out)
    levels = [f"L{i}" for i in range(10)]
    for row_i, q_i in enumerate(inds):
        qmeta = queries[int(q_i)]
        pos = [j for j, c in enumerate(candidates) if c["scene_key"] == qmeta["scene_key"]]
        neg = [j for j, c in enumerate(candidates) if c["scene_key"] != qmeta["scene_key"] and c["map_name"] == qmeta["map_name"]]
        if not pos:
            continue
        pos_j = max(pos, key=lambda j: sims[q_i, j])
        neg_j = max(neg, key=lambda j: sims[q_i, j]) if neg else int(rng.integers(len(candidates)))
        panels = [_open(root / qmeta["image_path"]), _open(root / candidates[pos_j]["image_path"])]
        panels += [Image.fromarray(make_levels(np.asarray(panels[1]), _masks(root, candidates[pos_j]))[level]) for level in levels[1:]]
        labels = ["Query RGB", "L0 Goal"] + levels[1:]
        y = row_i * panel_h
        instruction = qmeta.get("instruction", "")[:150]
        draw.text((2, y + 2), instruction, fill="black")
        for col, (im, label) in enumerate(zip(panels, labels)):
            im.thumbnail((panel_w, panel_h - 78))
            x = col * panel_w
            out.paste(im, (x, y + 56))
            draw.text((x + 3, y + 38), f"{label}", fill="black")
        draw.text((2, y + panel_h - 18),
                  f"{qmeta['distance_bin']} d={qmeta['distance_to_goal_m']:.1f}m | pos={sims[q_i,pos_j]:.3f} hardneg={sims[q_i,neg_j]:.3f} margin={sims[q_i,pos_j]-sims[q_i,neg_j]:.3f}",
                  fill="black")
    output_dir.mkdir(parents=True, exist_ok=True)
    out.save(output_dir / f"visual_examples_{split}_n{len(inds)}.jpg", quality=88)


def _save_full_gate_examples(root, encoder, split, extent, output_dir, count=100, batch_size=24):
    """Full-RGB-only audit panels; safe even when Gate 1 stops the ladder."""
    candidates = [r for r in _read_jsonl(root / f"templates_{split}.jsonl") if r["extent_m"] == extent]
    queries = _read_jsonl(root / f"queries_{split}.jsonl")
    if not queries or not candidates:
        return
    qf = _encode_paths(encoder, [_candidate_path(root, r) for r in queries], batch_size)
    cf = _encode_paths(encoder, [_candidate_path(root, r) for r in candidates], batch_size)
    qn = qf / np.maximum(np.linalg.norm(qf, axis=1, keepdims=True), 1e-8)
    cn = cf / np.maximum(np.linalg.norm(cf, axis=1, keepdims=True), 1e-8)
    sims = qn @ cn.T
    inds = np.linspace(0, len(queries) - 1, min(count, len(queries)), dtype=int)
    panel_w, panel_h = 224, 250
    out = Image.new("RGB", (panel_w * 3, panel_h * len(inds)), "white")
    draw = ImageDraw.Draw(out)
    for row_i, q_i in enumerate(inds):
        qmeta = queries[int(q_i)]
        pos = [j for j, c in enumerate(candidates) if c["scene_key"] == qmeta["scene_key"]]
        neg = [j for j, c in enumerate(candidates) if c["scene_key"] != qmeta["scene_key"] and c["map_name"] == qmeta["map_name"]]
        if not pos:
            continue
        pos_j = max(pos, key=lambda j: sims[q_i, j])
        neg_j = max(neg, key=lambda j: sims[q_i, j]) if neg else int(np.argmax(sims[q_i]))
        y = row_i * panel_h
        draw.text((2, y + 2), qmeta.get("instruction", "")[:105], fill="black")
        panels = [_open(root / qmeta["image_path"]), _open(root / candidates[pos_j]["image_path"]),
                  _open(root / candidates[neg_j]["image_path"])]
        labels = ["Trajectory-pose orthophoto query", "GT goal template", "Same-map hard negative"]
        for col, (im, label) in enumerate(zip(panels, labels)):
            im.thumbnail((panel_w, panel_h - 54))
            x = col * panel_w
            draw.text((x + 3, y + 25), label, fill="black")
            out.paste(im, (x, y + 43))
        draw.text((2, y + panel_h - 12),
                  f"{qmeta['distance_bin']} d={qmeta['distance_to_goal_m']:.1f}m | pos={sims[q_i,pos_j]:.3f} hardneg={sims[q_i,neg_j]:.3f} margin={sims[q_i,pos_j]-sims[q_i,neg_j]:.3f}",
                  fill="black")
    output_dir.mkdir(parents=True, exist_ok=True)
    out.save(output_dir / f"full_rgb_gate_examples_{split}_n{len(inds)}.jpg", quality=88)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=Path("../artifacts/visual_goal_abstraction/dataset_v2"))
    p.add_argument("--output-dir", type=Path, default=Path("../artifacts/visual_goal_abstraction/eval"))
    p.add_argument("--models", nargs="+", default=["siglip", "siglip2", "siglip2_partial", "dinov2_small"])
    p.add_argument("--siglip-model-path")
    p.add_argument("--siglip2-model-path", default="/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache/models--google--siglip2-so400m-patch16-512/snapshots/ceea1cba8130d8271436da4828633198c176a775")
    p.add_argument("--siglip2-partial-checkpoint", default="../artifacts/visual_goal_abstraction/models/siglip2_partial_trainseen.pt")
    p.add_argument("--dinov2-model-path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--max-queries", type=int, default=0, help="0 evaluates every query")
    p.add_argument("--abstraction", action="store_true", help="run L0-L9 only after the Full RGB gate")
    p.add_argument("--ablation", action="store_true", help="run single-factor ablations only after the Full RGB gate")
    args = p.parse_args()
    paths = {"siglip": args.siglip_model_path, "siglip2": args.siglip2_model_path,
             "siglip2_partial": args.siglip2_partial_checkpoint,
             "dinov2_small": args.dinov2_model_path}
    results = {"protocol": "offline goal imagery + real CityNav trajectory RGB; test_unseen prohibited",
               "models": {}, "view_limits": {"oblique": "unavailable: no calibrated oblique renderer",
                                              "FPV": "unavailable: no 3D model / camera geometry"}}
    for name in args.models:
        encoder = build_encoder(name, args.device, paths[name])
        results["models"][name] = {}
        for split in ("val_seen", "val_unseen"):
            candidates = _read_jsonl(args.dataset_dir / f"templates_{split}.jsonl")
            queries = _read_jsonl(args.dataset_dir / f"queries_{split}.jsonl")
            if args.max_queries:
                indices = np.linspace(0, len(queries) - 1, min(args.max_queries, len(queries)), dtype=int)
                queries = [queries[int(i)] for i in indices]
            split_result = {"by_extent": {}}
            qf = _encode_paths(encoder, [_candidate_path(args.dataset_dir, r) for r in queries], args.batch_size)
            for extent in (40.0, 80.0, 120.0):
                cands = [r for r in candidates if r["extent_m"] == extent]
                cf = _encode_paths(encoder, [_candidate_path(args.dataset_dir, r) for r in cands], args.batch_size)
                m = _dataset_metrics(qf, cf, queries, cands)
                sim = (qf / np.maximum(np.linalg.norm(qf, axis=1, keepdims=True), 1e-8)) @ (cf / np.maximum(np.linalg.norm(cf, axis=1, keepdims=True), 1e-8)).T
                pos = np.asarray([max(sim[i, [j for j, c in enumerate(cands) if c["scene_key"] == q["scene_key"]]], default=np.nan) for i, q in enumerate(queries)])
                m["distance_confounds"] = partial_distance_correlation(
                    [q["distance_to_goal_m"] for q in queries], pos,
                    [q["altitude_agl_m"] for q in queries], [q["brightness"] for q in queries],
                    [q["map_name"] for q in queries])
                split_result["by_extent"][str(int(extent))] = m
            # Model/scale are selected on val_seen only. No val_unseen threshold is fit.
            results["models"][name][split] = split_result
        seen = results["models"][name]["val_seen"]["by_extent"]
        extent_gates = {ext: _gate(metric) for ext, metric in seen.items()}
        best_extent = max(seen, key=lambda ext: (
            int(extent_gates[ext]["pass"]),
            extent_gates[ext]["same_map_accuracy"] or -1,
            extent_gates[ext]["same_map_margin"] or -1,
            -(extent_gates[ext]["distance_spearman_rho"] or 0),
        ))
        results["models"][name]["val_seen_selection"] = {"extent_m": int(best_extent), "gate": _gate(seen[best_extent])}
    selected = max(args.models, key=lambda n: (
        int(results["models"][n]["val_seen_selection"]["gate"].get("pass", False)),
        results["models"][n]["val_seen_selection"]["gate"].get("same_map_accuracy") or -1,
        results["models"][n]["val_seen_selection"]["gate"].get("same_map_margin") or -1,
    ))
    selected_extent = results["models"][selected]["val_seen_selection"]["extent_m"]
    results["selected_on_val_seen_only"] = {"encoder": selected, "extent_m": selected_extent}
    gate_seen = results["models"][selected]["val_seen_selection"]["gate"]
    heldout_metric = results["models"][selected]["val_unseen"]["by_extent"][str(selected_extent)]
    gate_unseen = _gate(heldout_metric)
    gate = {"pass": bool(gate_seen["pass"] and gate_unseen["pass"]),
            "val_seen": gate_seen, "val_unseen_heldout": gate_unseen,
            "selection_split": "val_seen", "val_unseen_used_for_selection": False}
    results["gate1"] = gate
    gate_encoder = build_encoder(selected, args.device, paths[selected])
    for split in ("val_seen", "val_unseen"):
        _save_full_gate_examples(args.dataset_dir, gate_encoder, split, selected_extent,
                                 args.output_dir / "visualizations", count=100,
                                 batch_size=args.batch_size)
    if gate["pass"] and args.abstraction:
        encoder = gate_encoder
        results["abstraction"] = {}
        for split in ("val_seen", "val_unseen"):
            levels = _score_abstraction(args.dataset_dir, encoder, split, selected_extent, args.batch_size)
            results["abstraction"][split] = levels
            _save_visual_grids(args.dataset_dir, encoder, split, selected_extent,
                               args.output_dir / "visualizations", count=100, batch_size=args.batch_size)
    if gate["pass"] and args.ablation:
        encoder = build_encoder(selected, args.device, paths[selected])
        results["controlled_ablation"] = {
            split: _score_controlled_ablation(args.dataset_dir, encoder, split, selected_extent, args.batch_size)
            for split in ("val_seen", "val_unseen")
        }
    else:
        results["abstraction_status"] = "STOPPED: Full RGB Gate 1 did not pass on val_seen" if not gate["pass"] else "not requested"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "full_goal_retrieval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"gate1": gate, "selected_encoder": selected, "selected_extent_m": selected_extent,
                      "abstraction_status": results.get("abstraction_status", "evaluated")}, indent=2))


if __name__ == "__main__":
    main()
