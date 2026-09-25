#!/usr/bin/env python3
"""Train candidate verification with visual patches for each named landmark."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.path_local_verifier import (
    LandmarkAwareCandidateVerifier, soft_distance_ranking_loss,
)
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.observation import cropclient
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE
from scripts.evaluate_path_local_pointcloud_teacher import (
    normalized_to_world, visible_in_official_crop,
)
from scripts.evaluate_path_local_rgb_evidence import (
    candidate_patches, reconstruct_observation_pose,
)
from scripts.train_path_local_candidate_verifier import rows_for_records
from scripts.train_relation_geometry_ranker import (
    corrected_landmark_geometry,
    matched_same_map_switch_indices,
    structured_relation_cues,
)
from scripts.train_rgb_candidate_ranker import encode_images, landmark_geometry, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", type=Path, default=Path(
        "runs/path_local_candidate_verifier/features_8192.pt"))
    parser.add_argument("--records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/expanded_train_records.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--cache", type=Path, default=Path(
        "runs/landmark_visual_verifier/features.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/landmark_visual_verifier"))
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-landmarks", type=int, default=9)
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--switch-loss-weight", type=float, default=0.2)
    parser.add_argument("--switch-margin", type=float, default=0.5)
    parser.add_argument(
        "--geometry-frame", choices=("raw", "audited"), default="raw",
        help=("Candidate/landmark coordinate frame. 'audited' applies the "
              "train-selected map-to-language rotate_270 transform."))
    parser.add_argument(
        "--explicit-relation-cues", action="store_true",
        help="Append deterministic per-landmark spatial-relation cues.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def slice_data(data, count):
    if not count:
        return data
    return {key: (value[:count] if isinstance(value, (torch.Tensor, list)) else value)
            for key, value in data.items()}


@torch.inference_mode()
def encode_landmark_patches(records, rows, wrong_rows, split, vision_dim,
                            encoder, processor, args, device):
    count, slots = len(records), args.max_landmarks
    image = torch.zeros(count, slots, vision_dim, dtype=torch.float16)
    wrong_image = torch.zeros_like(image)
    visible = torch.zeros(count, slots, dtype=torch.bool)
    wrong_visible = torch.zeros_like(visible)
    trajectories = load_mturk_trajectories(split, "all", 50.0)
    cropclient.load_image_cache()
    pending, pending_slots = [], []

    def flush():
        if not pending:
            return
        vectors = encode_images(encoder, processor, pending, device).half()
        for (kind, row_index, landmark_index), vector in zip(
                pending_slots, vectors):
            destination = image if kind == "correct" else wrong_image
            destination[row_index, landmark_index] = vector
        pending.clear(); pending_slots.clear()

    for row_index, (record, row, wrong) in enumerate(zip(records, rows, wrong_rows)):
        trajectory = trajectories[int(record["index"])]
        modes_world = np.stack([
            normalized_to_world(mode, record["map_name"])
            for mode in record["modes"]])
        pose = reconstruct_observation_pose(
            trajectory.start_pose, modes_world,
            record["observation_action_step"])
        rgb = cropclient.crop_image(
            record["map_name"], pose, (args.view_size, args.view_size), "rgb")
        for kind, source, validity in (
                ("correct", row, visible), ("wrong", wrong, wrong_visible)):
            centers = source.get("landmark_centers", [])[:slots]
            if not centers:
                continue
            world = np.stack([
                normalized_to_world(center, record["map_name"])
                for center in centers])
            patches = candidate_patches(
                rgb, pose, world, args.view_size, args.patch_size,
                args.patch_side_m)
            for landmark_index, (point, patch) in enumerate(zip(world, patches)):
                if visible_in_official_crop(pose, point):
                    validity[row_index, landmark_index] = True
                    pending.append(patch)
                    pending_slots.append((kind, row_index, landmark_index))
            if len(pending) >= args.feature_batch_size:
                flush()
        if (row_index + 1) % 256 == 0:
            print(f"{split} landmark RGB: {row_index + 1}/{count}", flush=True)
    flush()
    return image, visible, wrong_image, wrong_visible


def attach_geometry_frame(saved, records_all, args):
    """Rebuild geometry after cache loading so stale caches cannot hide the frame."""
    if args.geometry_frame == "raw":
        return saved
    for split, limit in (("train_seen", args.max_train),
                         ("val_seen", args.max_val)):
        records = records_all[split][:limit or None]
        rows = rows_for_records(args.candidate_dir, split, records)
        switched = matched_same_map_switch_indices(rows)
        geometry, valid = corrected_landmark_geometry(
            rows, args.max_landmarks)
        wrong_geometry, wrong_valid = corrected_landmark_geometry(
            rows, args.max_landmarks, switched)
        saved[split]["geometry"] = geometry.float()
        saved[split]["valid"] = valid
        saved[split]["wrong_geometry"] = wrong_geometry.float()
        saved[split]["wrong_valid"] = wrong_valid
    return saved


def attach_explicit_relation_cues(saved, records_all, args):
    """Ground relation words to explicit axes without using target labels."""
    if not args.explicit_relation_cues:
        return saved
    for split, limit in (("train_seen", args.max_train),
                         ("val_seen", args.max_val)):
        records = records_all[split][:limit or None]
        rows = rows_for_records(args.candidate_dir, split, records)
        switched = matched_same_map_switch_indices(rows)
        cues = structured_relation_cues(rows, args.max_landmarks)
        # Slice to the frozen BERT width so repeatedly loading an enriched
        # in-memory dictionary cannot append the same cues twice.
        saved[split]["relation"] = torch.cat(
            (saved[split]["relation"][..., :768].float(), cues), dim=-1)
        saved[split]["wrong_relation"] = torch.cat((
            saved[split]["wrong_relation"][..., :768].float(), cues[switched]),
            dim=-1)
    return saved


def load_or_encode(args, device):
    records_all = torch.load(args.records, map_location="cpu", weights_only=False)
    if args.cache.exists():
        saved = torch.load(args.cache, map_location="cpu", weights_only=False)
        saved = attach_geometry_frame(saved, records_all, args)
        return attach_explicit_relation_cues(saved, records_all, args)
    base = torch.load(args.base_features, map_location="cpu", weights_only=False)
    processor = AutoImageProcessor.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    saved = {}
    for split, limit in (("train_seen", args.max_train),
                         ("val_seen", args.max_val)):
        data = copy.copy(slice_data(base[split], limit))
        records = records_all[split][:limit or None]
        rows = rows_for_records(args.candidate_dir, split, records)
        switched = matched_same_map_switch_indices(rows)
        wrong_rows = [rows[index] for index in switched.tolist()]
        (landmark_image, visual_valid, wrong_image,
         wrong_visual_valid) = encode_landmark_patches(
            records, rows, wrong_rows, split, data["image"].shape[-1],
            encoder, processor, args, device)
        wrong_geometry, wrong_valid = landmark_geometry(rows, switched)
        data.update({
            "landmark_image": landmark_image,
            "landmark_visual_valid": visual_valid,
            "wrong_landmark_image": wrong_image,
            "wrong_landmark_visual_valid": wrong_visual_valid,
            "wrong_text": data["text"][switched],
            "wrong_relation": data["relation"][switched],
            "wrong_geometry": wrong_geometry,
            "wrong_valid": wrong_valid.bool(),
        })
        saved[split] = data
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.cache)
    saved = attach_geometry_frame(saved, records_all, args)
    return attach_explicit_relation_cues(saved, records_all, args)


def train_dataset(data):
    return TensorDataset(
        data["image"], data["landmark_image"], data["text"], data["geometry"],
        data["relation"], data["valid"], data["landmark_visual_valid"],
        data["prior"], data["semantic"], data["distance"], data["wrong_text"],
        data["wrong_geometry"], data["wrong_relation"], data["wrong_valid"],
        data["wrong_landmark_image"], data["wrong_landmark_visual_valid"])


@torch.inference_mode()
def predict(model, data, device, control="normal"):
    candidate, landmark = data["image"], data["landmark_image"]
    text, geometry, relation = data["text"], data["geometry"], data["relation"]
    valid, visual_valid = data["valid"], data["landmark_visual_valid"]
    if control == "swap_candidate_rgb":
        candidate = candidate[:, [1, 0]]
    elif control == "zero_landmark_rgb":
        landmark = torch.zeros_like(landmark); visual_valid = torch.zeros_like(visual_valid)
    elif control == "wrong_landmark_rgb":
        landmark = data["wrong_landmark_image"]
        visual_valid = data["wrong_landmark_visual_valid"]
    elif control == "wrong_text_only":
        text = data["wrong_text"]
    elif control == "wrong_relation_only":
        relation = data["wrong_relation"]
    elif control == "wrong_geometry_only":
        geometry = data["wrong_geometry"]
        valid = data["wrong_valid"]
    elif control == "same_map_wrong_instruction":
        landmark = data["wrong_landmark_image"]
        visual_valid = data["wrong_landmark_visual_valid"]
        text = data["wrong_text"]; geometry = data["wrong_geometry"]
        relation = data["wrong_relation"]; valid = data["wrong_valid"]
    dataset = TensorDataset(
        candidate, landmark, text, geometry, relation, valid, visual_valid,
        data["prior"], data["semantic"])
    output = []
    for batch in DataLoader(dataset, batch_size=128):
        output.append(model(*(value.to(device) for value in batch)).cpu())
    return torch.cat(output)


def train_model(train, val, args, device):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = LandmarkAwareCandidateVerifier(
        train["image"].shape[-1], relation_dim=train["relation"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in DataLoader(train_dataset(train), batch_size=128, shuffle=True):
            values = [value.to(device) for value in batch]
            normal = model(*values[:9])
            wrong = model(
                values[0], values[14], values[10], values[11], values[12],
                values[13], values[15], values[7], values[8])
            distance = values[9]
            ranking = soft_distance_ranking_loss(normal, distance)
            nearest = distance.argmin(-1)
            other = 1 - nearest
            normal_margin = (
                normal.gather(1, nearest[:, None]).squeeze(1)
                - normal.gather(1, other[:, None]).squeeze(1))
            wrong_margin = (
                wrong.gather(1, nearest[:, None]).squeeze(1)
                - wrong.gather(1, other[:, None]).squeeze(1))
            # Candidate selection depends on relative scores.  Penalizing only
            # an absolute score lets a wrong instruction shift both candidates
            # equally without changing the selected goal.  The counterfactual
            # must instead reduce the correct-vs-other ranking margin.
            discriminative = distance.le(20).sum(-1).eq(1).float()
            weight = 0.25 + 0.75 * discriminative
            contrast = F.softplus(
                args.switch_margin - normal_margin + wrong_margin)
            contrast = (contrast * weight).sum() / weight.sum().clamp_min(1)
            loss = ranking + args.switch_loss_weight * contrast
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval(); score = predict(model, val, device)
        value = metrics(val["distance"], score.argmax(-1))
        wrong_score = predict(
            model, val, device, "same_map_wrong_instruction")
        wrong_value = metrics(val["distance"], wrong_score.argmax(-1))
        control_gap = value["hit20"] - wrong_value["hit20"]
        coarse = metrics(
            val["distance"],
            (val["prior"] + 0.5 * val["semantic"]).argmax(-1))
        passes_gate = value["hit20"] >= coarse["hit20"] and control_gap >= 1.0
        key = (int(passes_gate), value["hit20"], control_gap,
               -value["mean_ne"], -epoch)
        if best is None or key > best[0]:
            best = key, copy.deepcopy(model.state_dict()), epoch, value
        print(f"epoch={epoch} hit20={value['hit20']:.3f} "
              f"wrong={wrong_value['hit20']:.3f} gap={control_gap:.3f} "
              f"gate={passes_gate} mean_ne={value['mean_ne']:.3f}", flush=True)
    model.load_state_dict(best[1]); return model.eval(), best[2], best[3]


def main():
    args = parse_args(); device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    saved = load_or_encode(args, device)
    model, epoch, selected = train_model(
        saved["train_seen"], saved["val_seen"], args, device)
    val = saved["val_seen"]; distance = val["distance"]
    score = predict(model, val, device)
    controls = {name: metrics(distance, predict(
        model, val, device, name).argmax(-1)) for name in (
            "swap_candidate_rgb", "zero_landmark_rgb", "wrong_landmark_rgb",
            "wrong_text_only", "wrong_relation_only", "wrong_geometry_only",
            "same_map_wrong_instruction")}
    coarse = metrics(
        distance, (val["prior"] + 0.5 * val["semantic"]).argmax(-1))
    language_gap = selected["hit20"] - controls[
        "same_map_wrong_instruction"]["hit20"]
    result = {
        "experiment": "candidate plus per-named-landmark visual verifier",
        "geometry_frame": args.geometry_frame,
        "explicit_relation_cues": args.explicit_relation_cues,
        "counts": {split: len(data["distance"]) for split, data in saved.items()},
        "selected_on_val_seen": {"epoch": epoch, **selected},
        "val_seen": {
            "coarse_semantic_fusion": coarse,
            "candidate_only_verifier_reference": None,
            "landmark_visual_verifier": metrics(distance, score.argmax(-1)),
            **controls,
            "two_mode_oracle": metrics(distance, distance.argmin(-1)),
        },
        "selection_gate": {
            "minimum_wrong_instruction_drop_pp": 1.0,
            "must_match_or_beat_coarse_semantic": True,
            "language_control_gap_pp": language_gap,
            "passed": (selected["hit20"] >= coarse["hit20"]
                       and language_gap >= 1.0),
        },
        "visibility": {
            split: {
                "visible_landmark_percent": float(
                    data["landmark_visual_valid"].sum()
                    / data["valid"].sum().clamp_min(1) * 100),
                "any_visible_episode_percent": float(
                    data["landmark_visual_valid"].any(-1).float().mean() * 100),
            } for split, data in saved.items()
        },
        "anti_leakage": {
            "observation_pose": "start pose + predicted modes only",
            "landmark_patches": "map-bound named landmark centers in current RGB",
            "wrong_instruction_control": (
                "same observation RGB + same-map wrong named landmark regions, "
                "text, relation features and geometry"),
            "target_use": "ranking loss and final metrics only",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "vision_dim": val["image"].shape[-1],
        "relation_dim": val["relation"].shape[-1],
        "epoch": epoch,
    }, args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
