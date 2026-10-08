#!/usr/bin/env python3
"""Light partial SigLIP2 tuning using only train_seen trajectory/template pairs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=Path("../artifacts/visual_goal_abstraction/dataset_v2"))
    p.add_argument("--model-path", default="/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache/models--google--siglip2-so400m-patch16-512/snapshots/ceea1cba8130d8271436da4828633198c176a775")
    p.add_argument("--output", type=Path, default=Path("../artifacts/visual_goal_abstraction/models/siglip2_partial_trainseen.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-pairs", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("partial SigLIP2 tuning requires CUDA; the base encoders can still be evaluated on CPU")
    root = args.dataset_dir
    queries = read_jsonl(root / "queries_train_seen.jsonl")
    validation_rows = read_jsonl(root / "queries_val_seen.jsonl")
    validation_scenes = {r["scene_key"] for r in validation_rows}
    def annotation_key(row):
        return row.get("annotation_key", f"{row['scene_key']}|{row.get('instruction', '')}")
    validation_annotations = {annotation_key(r) for r in validation_rows}
    templates = [r for r in read_jsonl(root / "templates_train_seen.jsonl") if r["extent_m"] == 80.0]
    by_scene = {}
    for row in templates:
        by_scene.setdefault(row["scene_key"], []).append(row)
    before_filter = len(queries)
    pairs = [q for q in queries if q["scene_key"] in by_scene
             and q["scene_key"] not in validation_scenes
             and annotation_key(q) not in validation_annotations]
    # One query per scene avoids treating duplicate annotations of one target as
    # in-batch negatives. All training pairs remain from train_seen.
    unique = {}
    for row in pairs:
        unique.setdefault(row["scene_key"], row)
    pairs = list(unique.values())
    random.shuffle(pairs)
    pairs = pairs[:args.max_pairs]
    if len(pairs) < args.batch_size * 2:
        raise RuntimeError(f"too few unique train_seen scenes ({len(pairs)})")

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device).train()
    for param in model.parameters():
        param.requires_grad_(False)
    trainable_prefixes = ("vision_model.encoder.layers.25.", "vision_model.encoder.layers.26.",
                          "vision_model.post_layernorm.", "vision_model.head.")
    trainable = []
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            param.requires_grad_(True); trainable.append(param)
    if not trainable:
        raise RuntimeError("could not locate the final two SigLIP2 vision blocks")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    log = []
    pairs_by_map = {}
    for row in pairs:
        pairs_by_map.setdefault(row["map_name"], []).append(row)
    for epoch in range(args.epochs):
        for group in pairs_by_map.values():
            random.shuffle(group)
        map_names = list(pairs_by_map)
        random.shuffle(map_names)
        batches = []
        for map_name in map_names:
            group = pairs_by_map[map_name]
            batches.extend(group[i:i + args.batch_size] for i in range(0, len(group) - args.batch_size + 1, args.batch_size))
        random.shuffle(batches)
        epoch_losses = []
        for batch in batches:
            # Keep each scene unique inside the contrastive batch.
            if len({x["scene_key"] for x in batch}) != len(batch):
                continue
            query_images = [Image.open(root / row["image_path"]).convert("RGB") for row in batch]
            goal_images = []
            for row in batch:
                cand = random.choice(by_scene[row["scene_key"]])
                goal_images.append(Image.open(root / cand["image_path"]).convert("RGB"))
            qin = processor(images=query_images, return_tensors="pt").to(device)
            gin = processor(images=goal_images, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                q = F.normalize(model.get_image_features(**qin).float(), dim=-1)
                g = F.normalize(model.get_image_features(**gin).float(), dim=-1)
                logits = (q @ g.T) * 20.0
                labels = torch.arange(len(batch), device=device)
                loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        summary = {"epoch": epoch + 1, "train_pairs": len(pairs), "pre_filter_queries": before_filter,
                   "excluded_val_seen_scene_count": len(validation_scenes),
                   "updates": len(epoch_losses),
                   "loss": float(np.mean(epoch_losses)) if epoch_losses else None}
        log.append(summary); print(json.dumps(summary), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tuned_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith(trainable_prefixes)}
    torch.save({"model_path": args.model_path, "trainable_prefixes": trainable_prefixes,
                "state_dict": tuned_state, "training": {"split": "train_seen", "epochs": args.epochs,
                "max_pairs": args.max_pairs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed},
                "log": log}, args.output)
    print(f"saved {len(tuned_state)} partial-tuned tensors to {args.output}")


if __name__ == "__main__":
    main()
