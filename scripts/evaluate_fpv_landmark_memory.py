#!/usr/bin/env python3
"""Frozen visual-language retrieval and bounded appearance-memory ablations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multiagent.fpv_landmark_memory.memory import AppearanceMemory, observation_quality
from multiagent.rsrefseg2_grounding.metrics import aggregate, retrieval_result


def crop_from_mask(image, mask, padding=0.12):
    y, x = np.where(np.asarray(mask) > 0)
    if not len(x):
        return None
    h, w = image.shape[:2]
    pad_x = int(max(x.max() - x.min() + 1, 1) * padding)
    pad_y = int(max(y.max() - y.min() + 1, 1) * padding)
    box = (max(0, x.min() - pad_x), max(0, y.min() - pad_y),
           min(w, x.max() + pad_x + 1), min(h, y.max() + pad_y + 1))
    return image[box[1]:box[3], box[0]:box[2]]


def quality_from_candidate(c, view_angle):
    return observation_quality(c.get("visible_ratio", 0), c.get("pixel_area", 0),
                               c.get("distance_m", 1e6), view_angle)


class Siglip2:
    def __init__(self, name):
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
        self.cache = {}
        self.text_cache = {}

    @torch.inference_mode()
    def image_features(self, images):
        values = []
        todo, indices = [], []
        for idx, (key, image) in enumerate(images):
            if key in self.cache:
                values.append((idx, self.cache[key]))
            else:
                indices.append(idx); todo.append((key, image))
        if todo:
            for start in range(0, len(todo), 32):
                chunk = todo[start:start + 32]
                pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for _, im in chunk]
                pixels = self.processor(images=pil, return_tensors="pt")["pixel_values"].cuda().to(torch.bfloat16)
                feat = self.model.get_image_features(pixel_values=pixels).float()
                feat = F.normalize(feat, dim=-1).cpu().numpy()
                for (key, _), vector in zip(chunk, feat):
                    self.cache[key] = vector
        return [self.cache[key] for key, _ in images]

    @torch.inference_mode()
    def text_feature(self, text):
        if text in self.text_cache:
            return self.text_cache[text]
        tok = self.processor(text=[text], padding=True, truncation=True, max_length=64, return_tensors="pt")
        tok = {k: v.cuda() for k, v in tok.items() if k in ("input_ids", "attention_mask")}
        value = F.normalize(self.model.get_text_features(**tok).float(), dim=-1)[0].cpu().numpy()
        self.text_cache[text] = value
        return value


class RSRef:
    def __init__(self, official_root, official_checkpoint, adapted_checkpoint=None):
        from multiagent.rsrefseg2_grounding.official_adapter import OfficialCoarseGrounder, configure_trainable
        model = OfficialCoarseGrounder(Path(official_root))
        model.load_official_checkpoint(Path(official_checkpoint))
        if adapted_checkpoint:
            state = torch.load(adapted_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=False)
        configure_trainable(model, "frozen")
        self.model = model.to("cuda", dtype=torch.bfloat16).eval()
        self.processor = model.processor
        self.cache = {}

    @torch.inference_mode()
    def crop_scores(self, text, keyed_images):
        out = []
        keyed_images = [(f"{text}\0{key}", image) for key, image in keyed_images]
        missing = [(key, im) for key, im in keyed_images if key not in self.cache]
        for start in range(0, len(missing), 8):
            chunk = missing[start:start + 8]
            imgs = []
            for _, im in chunk:
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
                imgs.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = self.model(torch.stack(imgs).cuda().to(torch.bfloat16), [text] * len(imgs))["coarse_logits"]
            for (key, _), logits in zip(chunk, prediction):
                scores = logits.float().flatten()
                self.cache[key] = float(scores.topk(max(1, scores.numel() // 20)).values.mean().cpu())
        for key, _ in keyed_images:
            out.append(float(self.cache.get(key, -float("inf"))))
        return out


def load_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def file_sha256(path):
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observations_for(row, view):
    if view == "topdown":
        yield {"step": row["step"], "offset": 0, "image": row["topdown_image"],
               "label_file": row["source_label_file"], "candidates": row["candidates"], "view_angle": 90.0}
    elif view in ("fpv", "oblique30", "oblique45"):
        yield {**row["view_observations"][view], "view_angle": 0.0 if view == "fpv" else (30.0 if view == "oblique30" else 45.0)}


def crops_and_quality(obs, candidate_ids, cache):
    path = obs["image"]
    image = load_rgb(path)
    payload = np.load(obs["label_file"])
    by_id = {int(c["landmark_id"]): c for c in obs["candidates"]}
    crops, qualities, visible = {}, {}, {}
    for oid in candidate_ids:
        c = by_id.get(oid)
        keyname = f"candidate_{oid}"
        if c is None or not c.get("visible", True) or keyname not in payload.files:
            crops[oid] = None; qualities[oid] = 0.0; visible[oid] = False
            continue
        crop = crop_from_mask(image, payload[keyname])
        crops[oid] = crop
        qualities[oid] = quality_from_candidate(c, obs.get("view_angle", 0.0))
        visible[oid] = crop is not None
    return crops, qualities, visible


def score_observation(model, text, obs, candidate_ids, crop_cache):
    crops, qualities, visible = crops_and_quality(obs, candidate_ids, crop_cache)
    ids = [x for x in candidate_ids if crops[x] is not None]
    scores = {x: -float("inf") for x in candidate_ids}
    features = {}
    if ids:
        keyed = [(f"{obs['image']}|{oid}", crops[oid]) for oid in ids]
        if isinstance(model, Siglip2):
            feats = model.image_features(keyed)
            textf = model.text_feature(text)
            features.update({oid: feat for oid, feat in zip(ids, feats)})
            scores.update({oid: float(np.dot(feat, textf)) for oid, feat in zip(ids, feats)})
        else:
            values = model.crop_scores(text, keyed)
            scores.update({oid: value for oid, value in zip(ids, values)})
    return scores, qualities, visible, features


def score_rank(row, id_scores):
    ids = [int(c["landmark_id"]) for c in row["candidates"]]
    vals = [float(id_scores.get(oid, -float("inf"))) for oid in ids]
    positives = set(map(int, row.get("referenced_landmark_ids", ())))
    has_positive = any(np.isfinite(id_scores.get(oid, -float("inf"))) for oid in positives)
    result = retrieval_result(row, vals) if has_positive else {
        "rank": len(ids) + 1, "top1": 0.0, "top4": 0.0, "top8": 0.0,
        "mrr": 1.0 / (len(ids) + 1), "margin": float("nan"),
        "localization_distance_m": float("inf"), "recall@20m": 0.0,
        "recall@40m": 0.0, "candidate_count": len(ids)}
    result["no_match"] = float(not has_positive)
    return result


def pool_views(view_scores, view_qualities, candidate_ids, method, n_views):
    chosen = list(range(min(int(n_views), len(view_scores))))
    out = {}
    for oid in candidate_ids:
        values = np.asarray([view_scores[i].get(oid, -float("inf")) for i in chosen], dtype=float)
        quality = np.asarray([view_qualities[i].get(oid, 0.0) for i in chosen], dtype=float)
        good = np.isfinite(values) & (quality > 0)
        if not good.any():
            out[oid] = -float("inf"); continue
        if method == "max":
            out[oid] = float(values[good].max())
        elif method == "quality":
            weights = quality[good] / quality[good].sum()
            out[oid] = float(np.sum(values[good] * weights))
        elif method == "mean":
            out[oid] = float(values[good].mean())
        else:
            raise ValueError(method)
    return out


def pool_feature_views(view_result, candidate_ids, n_views, method, text_feature):
    out = {}
    for oid in candidate_ids:
        vectors, weights = [], []
        for scores, quality, visibility, _obs, features in view_result[:n_views]:
            if visibility.get(oid, False) and oid in features:
                vectors.append(features[oid]); weights.append(max(float(quality.get(oid, 0)), 1e-8))
        if not vectors:
            out[oid] = -float("inf"); continue
        matrix = np.asarray(vectors, dtype=np.float32)
        if method == "max":
            out[oid] = float(np.max(matrix @ text_feature))
        elif method == "mean":
            pooled = matrix.mean(axis=0)
            pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
            out[oid] = float(np.dot(pooled, text_feature))
        elif method == "quality":
            weight = np.asarray(weights, dtype=np.float32); weight /= weight.sum()
            pooled = np.sum(matrix * weight[:, None], axis=0)
            pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
            out[oid] = float(np.dot(pooled, text_feature))
        else:
            raise ValueError(method)
    return out


def select_view_policy(view_result, candidate_ids, policy):
    out = {}
    for oid in candidate_ids:
        available = [(i, x) for i, x in enumerate(view_result) if x[2].get(oid, False)
                     and np.isfinite(x[0].get(oid, -float("inf")))]
        if not available:
            out[oid] = -float("inf"); continue
        if policy == "current":
            chosen = next((x for i, x in available if i == 0), available[0][1])
        elif policy == "closest":
            chosen = min((x for _, x in available), key=lambda x: next(c.get("distance_m", float("inf")) for c in x[3]["candidates"] if int(c["landmark_id"]) == oid))
        elif policy == "largest_visible_area":
            chosen = max((x for _, x in available), key=lambda x: next(c.get("pixel_area", 0) for c in x[3]["candidates"] if int(c["landmark_id"]) == oid))
        elif policy == "quality":
            chosen = max((x for _, x in available), key=lambda x: x[1].get(oid, 0))
        elif policy == "oracle_similarity":
            chosen = max((x for _, x in available), key=lambda x: x[0].get(oid, -float("inf")))
        else:
            raise ValueError(policy)
        out[oid] = float(chosen[0][oid])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True, help="AirSim FPV dataset manifest/output")
    p.add_argument("--source-dataset", type=Path, required=True, help="RSRefSeg2 top-down manifest")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("siglip2", "rsref_frozen", "rsref_prompter"), default="siglip2")
    p.add_argument("--model-name", default="google/siglip2-base-patch16-256")
    p.add_argument("--official-root", type=Path, default=Path("/mnt/windows-data/external/RSRefSeg2"))
    p.add_argument("--official-checkpoint", type=Path, default=Path("/mnt/windows-data/hett-rsrefseg2/external/refsegrs.pth"))
    p.add_argument("--prompter-checkpoint", type=Path, default=Path("/mnt/windows-data/hett-rsrefseg2/runs/fixedpad_experiment_b_prompter_s0/artifacts/best.pt"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--language", choices=("full", "name_category", "referenced_phrase"), default="full")
    a = p.parse_args()
    if "test_unseen" in " ".join(map(str, vars(a).values())):
        raise ValueError("test_unseen is forbidden")
    started = time.time(); a.output.mkdir(parents=True, exist_ok=True)
    src = {json.loads(line)["sample_id"]: json.loads(line) for line in (a.source_dataset / "manifest.jsonl").read_text().splitlines()}
    rows = [json.loads(line) for line in (a.dataset / "manifest.jsonl").read_text().splitlines()]
    allowed_splits = {"train_seen", "val_seen", "val_unseen"}
    if any(row.get("split") not in allowed_splits for row in rows) or any(row.get("split") not in allowed_splits for row in src.values()):
        raise ValueError("unexpected split in grounding dataset; test_unseen is forbidden")
    rows = rows[:a.limit or None]
    if a.method == "siglip2":
        model = Siglip2(a.model_name)
        selected_model = model.model
    else:
        adapted = a.prompter_checkpoint if a.method == "rsref_prompter" else None
        model = RSRef(a.official_root, a.official_checkpoint, adapted)
        selected_model = model.model
    parameter_metadata = {"total_parameters": int(sum(p.numel() for p in selected_model.parameters())),
                          "trainable_parameters": int(sum(p.numel() for p in selected_model.parameters() if p.requires_grad))}
    records = []
    for row in rows:
        source = src[row["sample_id"]]
        row["source_label_file"] = str((a.source_dataset / source["label_file"]).resolve()) if not Path(source["label_file"]).is_absolute() else source["label_file"]
        row["topdown_image"] = source["image"]
        row["candidates"] = source["candidates"]
        row["referenced_landmark_ids"] = source["referenced_landmark_ids"]
        row["positives"] = source.get("positives", [])
        row["instruction"] = source["instruction"]
        categories = sorted({x.get("category", "") for x in row.get("positives", []) if x.get("category")})
        name_category = (", ".join(row["referenced_landmark_names"]) + " " + ", ".join(categories)).strip()
        text = row["instruction"] if a.language == "full" else (
            name_category if a.language == "name_category" else row.get("referenced_phrase", ""))
        ids = [int(c["landmark_id"]) for c in row["candidates"]]
        ref_ids = set(map(int, row["referenced_landmark_ids"]))
        positive_meta = next((x for x in row["positives"] if int(x["landmark_id"]) in ref_ids), {})
        positive_category = positive_meta.get("category", "unknown")
        positive_xy = np.asarray(positive_meta.get("world_xy", [np.nan, np.nan]), dtype=float)
        current_xy = np.asarray(row["view_observations"]["fpv"]["pose"][:2], dtype=float)
        reference_distance = float(np.linalg.norm(positive_xy - current_xy)) if np.isfinite(positive_xy).all() else float("nan")
        same_class = sum(1 for c in row["candidates"] if c.get("category") == positive_category and int(c["landmark_id"]) not in ref_ids)
        for view in ("topdown", "fpv", "oblique30", "oblique45"):
            obs = next(observations_for(row, view))
            scored, quality, visibility, _features = score_observation(model, text, obs, ids, {})
            records.append({"sample_id": row["sample_id"], "episode_id": row["episode_id"], "split": row["split"],
                            "view": view, "method": a.method, "language": a.language,
                            "metrics": score_rank(row, scored), "candidate_scores": scored,
                            "candidate_quality": quality, "candidate_visibility": visibility,
                            "target_current_visible": any(visibility.get(int(x), False) for x in row["referenced_landmark_ids"]),
                            "category": positive_category, "reference_distance_bucket": "<50m" if reference_distance < 50 else "50-100m" if reference_distance < 100 else "100-200m" if reference_distance < 200 else ">200m",
                            "candidate_count": len(ids), "same_class_distractors": same_class})
        history = row["history_fpv"]
        view_result = []
        for obs in history:
            scored, quality, visibility, features = score_observation(model, text, obs, ids, {})
            view_result.append((scored, quality, visibility, obs, features))
        primary = int(row["referenced_landmark_ids"][0]) if row["referenced_landmark_ids"] else -1
        current_vis = bool(view_result and view_result[0][2].get(primary, False))
        prior_vis = any(v[2].get(primary, False) for v in view_result[1:])
        status = "current_visible" if current_vis else ("current_invisible_previously_seen" if prior_vis else "never_seen")
        for n in (1, 3, 5, 8):
            if n > len(view_result):
                continue
            for aggregation in ("mean", "max", "quality"):
                if isinstance(model, Siglip2):
                    pooled = pool_feature_views(view_result, ids, n, aggregation, model.text_feature(text))
                else:
                    pooled = pool_views([x[0] for x in view_result], [x[1] for x in view_result], ids, aggregation, n)
                records.append({"sample_id": row["sample_id"], "episode_id": row["episode_id"], "split": row["split"],
                                "view": "fpv_memory", "method": a.method, "language": a.language,
                                "history_n": n, "aggregation": aggregation, "visibility_status": status,
                                "metrics": score_rank(row, pooled), "candidate_scores": pooled,
                                "target_current_visible": current_vis, "target_previously_seen": prior_vis,
                                "category": positive_category, "reference_distance_bucket": "<50m" if reference_distance < 50 else "50-100m" if reference_distance < 100 else "100-200m" if reference_distance < 200 else ">200m",
                                "candidate_count": len(ids), "same_class_distractors": same_class})
            past = view_result[1:]
            if len(past) >= n:
                if isinstance(model, Siglip2):
                    past_score = pool_feature_views(past, ids, n, "quality", model.text_feature(text))
                else:
                    past_score = pool_views([x[0] for x in past], [x[1] for x in past], ids, "quality", n)
                records.append({"sample_id": row["sample_id"], "episode_id": row["episode_id"], "split": row["split"],
                                "view": "memory_only", "method": a.method, "language": a.language,
                                "history_n": n, "aggregation": "quality", "visibility_status": status,
                                "metrics": score_rank(row, past_score), "candidate_scores": past_score,
                                "target_current_visible": current_vis, "target_previously_seen": prior_vis,
                                "category": positive_category, "reference_distance_bucket": "<50m" if reference_distance < 50 else "50-100m" if reference_distance < 100 else "100-200m" if reference_distance < 200 else ">200m",
                                "candidate_count": len(ids), "same_class_distractors": same_class})
        for policy in ("current", "closest", "largest_visible_area", "quality", "oracle_similarity"):
            chosen_scores = select_view_policy(view_result, ids, policy)
            records.append({"sample_id": row["sample_id"], "episode_id": row["episode_id"], "split": row["split"],
                            "view": f"view_selection_{policy}", "method": a.method, "language": a.language,
                            "visibility_status": status, "metrics": score_rank(row, chosen_scores),
                            "candidate_scores": chosen_scores, "target_current_visible": current_vis,
                            "target_previously_seen": prior_vis, "category": positive_category,
                            "reference_distance_bucket": "<50m" if reference_distance < 50 else "50-100m" if reference_distance < 100 else "100-200m" if reference_distance < 200 else ">200m",
                            "candidate_count": len(ids), "same_class_distractors": same_class})
        # Candidate-keyed landmark memory: retain the best 5 observations by fixed quality score.
        memory = AppearanceMemory(max_views_per_landmark=5)
        for scores, quality, visibility, obs, features in view_result:
            candidate_meta = {int(c["landmark_id"]): c for c in obs["candidates"]}
            for oid in ids:
                if not visibility.get(oid, False):
                    continue
                c = candidate_meta.get(oid, {})
                stored_feature = features.get(oid) if isinstance(model, Siglip2) else scores.get(oid, -float("inf"))
                if stored_feature is None:
                    continue
                memory.update(str(oid), stored_feature, step=int(obs["step"]),
                              visible_ratio=c.get("visible_ratio", 0), pixel_area=c.get("pixel_area", 0),
                              distance_m=c.get("distance_m", 1e6), view_angle_deg=0.0,
                              map_position=c.get("world_xy"), category_or_name=c.get("name", c.get("category", "")))
        memory_scores = {}
        for oid in ids:
            state = memory.get(str(oid))
            values = state.observations if state else []
            if not values:
                memory_scores[oid] = -float("inf")
                continue
            if isinstance(model, Siglip2):
                vectors = np.stack([np.asarray(x.feature, dtype=np.float32) for x in values])
                weights = np.asarray([max(x.quality, 1e-8) for x in values], dtype=np.float32)
                pooled = np.average(vectors, axis=0, weights=weights)
                pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
                memory_scores[oid] = float(np.dot(pooled, model.text_feature(text)))
            else:
                vals = np.asarray([float(x.feature) for x in values], float)
                weights = np.asarray([max(x.quality, 1e-8) for x in values], float)
                memory_scores[oid] = float(np.average(vals, weights=weights))
        records.append({"sample_id": row["sample_id"], "episode_id": row["episode_id"], "split": row["split"],
                        "view": "landmark_memory_top5", "method": a.method, "language": a.language,
                        "visibility_status": status, "metrics": score_rank(row, memory_scores),
                        "candidate_scores": memory_scores, "memory_count": len(memory.landmarks),
                        "landmark_memory": {str(oid): {"observation_count": len(memory.get(str(oid)).observations),
                                                        "first_seen_step": memory.get(str(oid)).first_seen_step,
                                                        "last_seen_step": memory.get(str(oid)).last_seen_step,
                                                        "best_visible_ratio": memory.get(str(oid)).best.visible_ratio,
                                                        "best_distance_m": memory.get(str(oid)).best.distance_m,
                                                        "best_view_angle_deg": 0.0}
                                             for oid in ids if memory.get(str(oid)) and memory.get(str(oid)).best},
                        "target_current_visible": current_vis, "target_previously_seen": prior_vis,
                        "category": positive_category, "reference_distance_bucket": "<50m" if reference_distance < 50 else "50-100m" if reference_distance < 100 else "100-200m" if reference_distance < 200 else ">200m",
                        "candidate_count": len(ids), "same_class_distractors": same_class})
        if len(records) % 100 == 0:
            print(f"evaluated {row['split']} {row['sample_id']} rows={len(records)}", flush=True)
    # Aggregate by split/view/memory configuration; the val_unseen split is only evaluated.
    groups = defaultdict(list)
    for rec in records:
        config = (rec["split"], rec["view"], rec.get("history_n", 0), rec.get("aggregation", ""), rec.get("visibility_status", ""))
        groups[config].append(rec["metrics"])
    summary = {"method": a.method, "language": a.language, "samples": len(rows),
               "splits": {s: {} for s in ("train_seen", "val_seen", "val_unseen")},
               "configs": {}, "seconds": time.time() - started,
               "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
               "parameter_metadata": parameter_metadata,
               "checkpoint_provenance": {"model_name": a.model_name if a.method == "siglip2" else "KyanChen/RSRefSeg2",
                                         "official_checkpoint": str(a.official_checkpoint) if a.method != "siglip2" else None,
                                         "official_checkpoint_sha256": file_sha256(a.official_checkpoint) if a.method != "siglip2" else None,
                                         "prompter_checkpoint": str(a.prompter_checkpoint) if a.method == "rsref_prompter" else None,
                                         "prompter_checkpoint_sha256": file_sha256(a.prompter_checkpoint) if a.method == "rsref_prompter" else None,
                                         "adaptation_trained_in_this_experiment": False},
               "gt_mask_model_input": False, "test_unseen_read": False,
               "history_selection": "current then raw-trajectory offsets 1,3,5,7,9,11; fixed quality weighting; no val_unseen tuning"}
    for key, values in groups.items():
        split, view, n, agg, status = key
        agg_values = aggregate(values)
        summary["configs"]["|".join(map(str, key))] = agg_values
    for split in summary["splits"]:
        for view in sorted({r["view"] for r in records if r["split"] == split}):
            summary["splits"][split][view] = aggregate([r["metrics"] for r in records if r["split"] == split and r["view"] == view])
    slices = defaultdict(list)
    for rec in records:
        key = (rec["split"], rec["view"], rec.get("history_n", 0), rec.get("aggregation", ""))
        for field in ("visibility_status", "category", "reference_distance_bucket"):
            if rec.get(field) is not None:
                slices[(*key, field, str(rec[field]))].append(rec["metrics"])
        if rec.get("candidate_count") is not None:
            n = int(rec["candidate_count"]); slices[(*key, "candidate_count", "<=4" if n <= 4 else "5-8" if n <= 8 else ">=9")].append(rec["metrics"])
            n = int(rec.get("same_class_distractors", 0)); slices[(*key, "same_class_distractors", "0" if n == 0 else "1-3" if n <= 3 else ">=4")].append(rec["metrics"])
    summary["slices"] = {"|".join(map(str, key)): aggregate(values) for key, values in slices.items()}
    (a.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    with (a.output / "per_sample.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, allow_nan=True) + "\n")
    with (a.output / "metrics.csv").open("w") as f:
        writer = csv.writer(f); writer.writerow(["split", "config", "samples", "top1", "top4", "mrr", "margin", "r20", "r40", "mean_distance"])
        for key, values in groups.items():
            m = aggregate(values); writer.writerow([key[0], "|".join(map(str, key)), m.get("samples", 0), m.get("top1"), m.get("top4"), m.get("mrr"), m.get("margin"), m.get("recall@20m"), m.get("recall@40m"), m.get("mean_localization_distance_m")])
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
