#!/usr/bin/env python3
"""Train an instance-level verifier on target-free path-local RGB patches.

The observation pose and candidate crops are functions only of predicted belief
modes and the start pose.  Reference targets create ranking labels and metrics;
they are excluded from the deployable feature dictionary and model signature.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.path_local_verifier import (
    PathLocalCandidateVerifier, soft_distance_ranking_loss,
)
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.observation import cropclient
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE, load_rows
from scripts.evaluate_path_local_pointcloud_teacher import normalized_to_world
from scripts.evaluate_path_local_rgb_evidence import (
    candidate_patches, reconstruct_observation_pose,
)
from scripts.train_rgb_candidate_ranker import (
    encode_images, landmark_geometry, metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/records.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--text-features", type=Path, default=DEFAULT_SOURCE /
                        "target_type_alignment/features.pt")
    parser.add_argument("--cache", type=Path, default=Path(
        "runs/path_local_candidate_verifier/features.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/path_local_candidate_verifier"))
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--switch-loss-weight", type=float, default=0.2)
    parser.add_argument("--switch-margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def rows_for_records(candidate_dir, split, records):
    rows = load_rows(candidate_dir / f"{split}.jsonl")
    output = []
    for record in records:
        row = copy.deepcopy(rows[int(record["index"])])
        row["candidates"] = np.asarray(record["modes"]).tolist()
        output.append(row)
    return output


def same_map_switch(map_names):
    grouped = defaultdict(list)
    for index, name in enumerate(map_names):
        grouped[name].append(index)
    output = []
    for index, name in enumerate(map_names):
        values = grouped[name]
        position = values.index(index)
        output.append(values[(position + 1) % len(values)])
    return torch.tensor(output, dtype=torch.long)


@torch.inference_mode()
def encode_path_patches(records, split, encoder, processor, args, device):
    trajectories = load_mturk_trajectories(split, "all", 50.0)
    image = None; pending = []; slots = []
    cropclient.load_image_cache()

    def flush():
        nonlocal image
        if not pending:
            return
        vector = encode_images(encoder, processor, pending, device)
        if image is None:
            image = torch.empty(len(records), 2, vector.shape[-1])
        for slot, value in zip(slots, vector):
            image[slot] = value
        pending.clear(); slots.clear()

    for row_index, record in enumerate(records):
        trajectory = trajectories[int(record["index"])]
        if trajectory.map_name != record["map_name"]:
            raise ValueError(f"trajectory mismatch at {split}/{record['index']}")
        modes_world = np.stack([
            normalized_to_world(mode, record["map_name"])
            for mode in record["modes"]])
        pose = reconstruct_observation_pose(
            trajectory.start_pose, modes_world,
            record["observation_action_step"])
        rgb = cropclient.crop_image(
            record["map_name"], pose, (args.view_size, args.view_size), "rgb")
        patches = candidate_patches(
            rgb, pose, modes_world, args.view_size, args.patch_size,
            args.patch_side_m)
        for candidate_index, patch in enumerate(patches):
            pending.append(patch); slots.append((row_index, candidate_index))
        if len(pending) >= args.feature_batch_size:
            flush()
        if (row_index + 1) % 256 == 0:
            print(f"{split} RGB: {row_index + 1}/{len(records)}", flush=True)
    flush()
    return image.half()


def build_split(records, rows, image, text, relation):
    geometry, valid = landmark_geometry(rows)
    modes = torch.from_numpy(np.stack([record["modes"] for record in records]))
    target = torch.from_numpy(np.stack([record["target"] for record in records]))
    result = {
        "image": image,
        "text": text,
        "geometry": geometry,
        "relation": relation,
        "valid": valid.bool(),
        "prior": torch.tensor(np.stack([record["prior"] for record in records])),
        "semantic": torch.tensor(np.stack([record["evidence"] for record in records])),
        "distance": torch.linalg.vector_norm(
            modes - target[:, None], dim=-1) * 410.0,
        "map_names": [record["map_name"] for record in records],
    }
    attach_switched_features(result, rows)
    return result


def attach_switched_features(data, rows):
    switched = same_map_switch(data["map_names"])
    switched_geometry, switched_valid = landmark_geometry(rows, switched)
    data["switched_text"] = data["text"][switched]
    data["switched_relation"] = data["relation"][switched]
    data["switched_geometry"] = switched_geometry
    data["switched_valid"] = switched_valid.bool()


def load_or_encode(args, device):
    if args.cache.exists():
        saved = torch.load(args.cache, map_location="cpu", weights_only=False)
        if set(saved) != {"train_seen", "val_seen"}:
            raise ValueError("feature cache must contain only seen splits")
        if "switched_geometry" not in saved["train_seen"]:
            records = torch.load(
                args.geometry_records, map_location="cpu", weights_only=False)
            for split in ("train_seen", "val_seen"):
                count = len(saved[split]["distance"])
                selected = records[split][:count]
                rows = rows_for_records(args.candidate_dir, split, selected)
                attach_switched_features(saved[split], rows)
            torch.save(saved, args.cache)
        return saved
    records = torch.load(
        args.geometry_records, map_location="cpu", weights_only=False)
    selected = {
        "train_seen": records["train_seen"][:args.max_train or None],
        "val_seen": records["val_seen"][:args.max_val or None],
    }
    rows = {split: rows_for_records(args.candidate_dir, split, values)
            for split, values in selected.items()}
    relation_all = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True)
    text_all = torch.load(
        args.text_features, map_location="cpu", weights_only=True)["text"]
    processor = AutoImageProcessor.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    saved = {}
    for split in ("train_seen", "val_seen"):
        indices = [int(record["index"]) for record in selected[split]]
        image = encode_path_patches(
            selected[split], split, encoder, processor, args, device)
        saved[split] = build_split(
            selected[split], rows[split], image,
            text_all[split]["features"][indices], relation_all[split][indices])
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.cache)
    del encoder
    torch.cuda.empty_cache()
    return saved


def dataset(data):
    return TensorDataset(
        data["image"], data["text"], data["geometry"], data["relation"],
        data["valid"], data["prior"], data["semantic"], data["distance"],
        data["switched_text"], data["switched_geometry"],
        data["switched_relation"], data["switched_valid"])


@torch.inference_mode()
def predict(model, data, device, control="normal"):
    image, text, relation = data["image"], data["text"], data["relation"]
    geometry, valid = data["geometry"], data["valid"]
    prior, semantic = data["prior"], data["semantic"]
    if control == "swap_candidate_rgb":
        image = image[:, [1, 0]]
    elif control == "zero_candidate_rgb":
        image = torch.zeros_like(image)
    elif control == "same_map_wrong_instruction":
        text = data["switched_text"]
        relation = data["switched_relation"]
        geometry = data["switched_geometry"]
        valid = data["switched_valid"]
    output = []
    values = TensorDataset(image, text, geometry, relation, valid, prior, semantic)
    for batch in DataLoader(values, batch_size=256):
        output.append(model(*(item.to(device) for item in batch)).cpu())
    return torch.cat(output)


def train_model(train, val, args, device):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = PathLocalCandidateVerifier(train["image"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in DataLoader(dataset(train), batch_size=256, shuffle=True):
            (image, text, geometry, relation, valid, prior, semantic, distance,
             switched_text, switched_geometry, switched_relation,
             switched_valid) = [value.to(device) for value in batch]
            score = model(
                image, text, geometry, relation, valid, prior, semantic)
            switched_score = model(
                image, switched_text, switched_geometry, switched_relation,
                switched_valid, prior, semantic)
            ranking_loss = soft_distance_ranking_loss(score, distance)
            nearest = distance.argmin(-1)
            correct = score.gather(1, nearest[:, None]).squeeze(1)
            wrong = switched_score.gather(1, nearest[:, None]).squeeze(1)
            switch_loss = torch.nn.functional.softplus(
                args.switch_margin - correct + wrong).mean()
            loss = ranking_loss + args.switch_loss_weight * switch_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval(); score = predict(model, val, device)
        value = metrics(val["distance"], score.argmax(-1))
        key = value["hit20"], -value["mean_ne"], -epoch
        if best is None or key > best[0]:
            best = key, copy.deepcopy(model.state_dict()), epoch, value
        print(f"epoch={epoch} hit20={value['hit20']:.3f} "
              f"mean_ne={value['mean_ne']:.3f}", flush=True)
    model.load_state_dict(best[1]); return model.eval(), best[2], best[3]


def centered_scores(score):
    value = score - score.mean(-1, keepdim=True)
    return value / value.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)


@torch.inference_mode()
def tune_fusion_weight(model, train, device):
    verifier = centered_scores(predict(model, train, device))
    base = train["prior"] + 0.5 * train["semantic"]
    best = None
    for weight in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        value = metrics(train["distance"], (base + weight * verifier).argmax(-1))
        key = value["hit20"], -value["mean_ne"], -weight
        if best is None or key > best[0]:
            best = key, weight, value
    return best[1], best[2]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = load_or_encode(args, device)
    model, epoch, selected = train_model(
        saved["train_seen"], saved["val_seen"], args, device)
    val = saved["val_seen"]; distance = val["distance"]
    score = predict(model, val, device)
    fusion_weight, train_fusion = tune_fusion_weight(
        model, saved["train_seen"], device)
    semantic_score = val["prior"] + 0.5 * val["semantic"]
    semantic_rank = semantic_score.argmax(-1)
    fused_score = semantic_score + fusion_weight * centered_scores(score)
    controls = {}
    for name in ("swap_candidate_rgb", "zero_candidate_rgb",
                 "same_map_wrong_instruction"):
        controlled = predict(model, val, device, name)
        controls[name] = metrics(
            distance,
            (semantic_score + fusion_weight * centered_scores(controlled)).argmax(-1))
    result = {
        "experiment": "path-local instance and relation candidate verifier",
        "counts": {split: len(value["distance"]) for split, value in saved.items()},
        "selected_on_val_seen": {"epoch": epoch, **selected},
        "selected_on_train_seen": {
            "fusion_weight": fusion_weight, "fusion_metrics": train_fusion},
        "val_seen": {
            "static_top1": metrics(
                distance, torch.zeros(len(distance), dtype=torch.long)),
            "coarse_semantic_fusion": metrics(distance, semantic_rank),
            "instance_relation_verifier_raw": metrics(
                distance, score.argmax(-1)),
            "coarse_plus_instance_verifier": metrics(
                distance, fused_score.argmax(-1)),
            **controls,
            "two_mode_oracle": metrics(distance, distance.argmin(-1)),
        },
        "anti_leakage": {
            "observation_pose": "start pose + predicted modes only",
            "runtime_inputs": (
                "path-local RGB patches, full instruction embedding, per-landmark "
                "relation geometry, belief prior, coarse semantic evidence"),
            "target_use": "ranking loss and final metrics only",
            "candidate_order_encoding": False,
            "injected_ground_truth_candidate": False,
            "val_unseen_loaded": False, "test_unseen_loaded": False,
        },
        "scope": "offline candidate-verification gate before closed-loop promotion",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "vision_dim": val["image"].shape[-1],
        "epoch": epoch,
    }, args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
