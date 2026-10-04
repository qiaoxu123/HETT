#!/usr/bin/env python3
"""SigLIP/SigLIP2 zero-shot prompt sensitivity on held-out objects."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from multiagent.visual_attributes.dataset import load_representation  # noqa: E402
from multiagent.visual_attributes.metrics import classification_metrics, grouped_consistency  # noqa: E402


PROMPTS = {
    "generic": "a {label} {noun}",
    "aerial": "an aerial view of a {label} {noun}",
    "drone": "a {label} {noun} viewed from a drone",
    "roof": "a {noun} with a {label} roof viewed from above",
}


def label_text(label, category):
    if category == "road_context":
        return "next to a road" if label == "yes" else "away from a road"
    if category == "roof_presence":
        return "visible roof" if label == "yes" else "no visible roof"
    return label.replace("_", " ")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="google/siglip2-base-patch16-256"); parser.add_argument("--representation", default="crop", choices=("whole", "crop", "masked"))
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--categories", nargs="+", default=("color", "size", "shape", "semantic", "road_context", "roof_presence"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModel, AutoProcessor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); processor = AutoProcessor.from_pretrained(args.model_name); model = AutoModel.from_pretrained(args.model_name).to(device).eval().requires_grad_(False)
    base = torch.load(args.dataset / "handcrafted_features.pt", map_location="cpu", weights_only=False); rows = base["rows"]
    selected_rows = [row for row in rows if row["split"] in ("val_seen", "val_unseen")]
    image_features = []
    started = time.time()
    for start in range(0, len(selected_rows), args.batch_size):
        images = [load_representation(args.dataset, row, args.representation) for row in selected_rows[start:start + args.batch_size]]
        inputs = processor(images=images, return_tensors="pt"); pixels = inputs["pixel_values"].to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            feature = model.get_image_features(pixel_values=pixels); feature = feature / feature.norm(dim=-1, keepdim=True)
        image_features.append(feature.float().cpu())
    image_features = torch.cat(image_features)
    result = {"model": args.model_name, "representation": args.representation, "parameters": sum(p.numel() for p in model.parameters()), "image_seconds": time.time() - started,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0, "categories": {}}
    for category in args.categories:
        classes = list(base["classes"][category]); noun = "object" if category in ("color", "size", "shape") else "scene"
        result["categories"][category] = {}
        for prompt_name, template in PROMPTS.items():
            prompts = [template.format(label=label_text(label, category), noun=noun) for label in classes]
            tokens = processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt"); tokens = {k: v.to(device) for k, v in tokens.items() if k in ("input_ids", "attention_mask", "pixel_attention_mask", "spatial_shapes")}
            with torch.inference_mode(): text = model.get_text_features(**tokens); text = text / text.norm(dim=-1, keepdim=True)
            probabilities = torch.softmax(image_features @ text.float().cpu().T, -1).numpy()
            prompt_result = {}
            for split in ("val_seen", "val_unseen"):
                indices = [i for i, row in enumerate(selected_rows) if row["split"] == split and row["labels"].get(category) in classes]
                targets = np.asarray([classes.index(selected_rows[i]["labels"][category]) for i in indices])
                metrics = classification_metrics(probabilities[indices], targets)
                metrics["cross_height"] = grouped_consistency(probabilities[indices], [selected_rows[i]["object_key"] for i in indices], [selected_rows[i]["height_m"] for i in indices])
                prompt_result[split] = metrics
            result["categories"][category][prompt_name] = prompt_result
    (args.output / "zero_shot_metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__": main()
