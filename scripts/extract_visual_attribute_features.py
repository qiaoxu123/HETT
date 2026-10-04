#!/usr/bin/env python3
"""Cache current-view features for fair frozen-encoder probes."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from multiagent.models.dark_net import Darknet  # noqa: E402
from multiagent.visual_attributes.dataset import load_representation  # noqa: E402


def darknet(args, device):
    model = Darknet(str(ROOT / "weights/yolo_v3.cfg"), 224).to(device).eval().requires_grad_(False)
    saved = torch.load(ROOT / "weights/best.pt", map_location="cpu", weights_only=False)["model"]; state = model.state_dict(); state.update({k: v for k, v in saved.items() if k in state}); model.load_state_dict(state)
    def encode(images):
        values = np.stack([cv2.resize(image, (224, 224)) for image in images])[:, :, :, ::-1].transpose(0, 3, 1, 2).copy().astype(np.float32)
        values = (values - np.asarray([60.134, 49.697, 40.746], np.float32).reshape(1, 3, 1, 1)) / np.asarray([29.99, 24.498, 22.046], np.float32).reshape(1, 3, 1, 1)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"): return model(torch.from_numpy(values).to(device)).mean((2, 3)).float().cpu()
    return model, encode


def huggingface(args, device):
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(args.model_name); model = AutoModel.from_pretrained(args.model_name).to(device).eval().requires_grad_(False)
    def encode(images):
        values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if hasattr(model, "get_image_features"):
                result = model.get_image_features(pixel_values=values)
            else:
                output = model(pixel_values=values)
                result = output.image_embeds if hasattr(output, "image_embeds") and output.image_embeds is not None else output.last_hidden_state[:, 0]
        return result.float().cpu()
    return model, encode


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("darknet", "dinov2", "siglip", "siglip2"), required=True); parser.add_argument("--model-name")
    parser.add_argument("--representations", nargs="+", default=("whole", "crop", "masked")); parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines()]
    if args.encoder == "dinov2" and not args.model_name: args.model_name = "facebook/dinov2-small"
    if args.encoder == "siglip" and not args.model_name: args.model_name = "google/siglip-base-patch16-224"
    if args.encoder == "siglip2" and not args.model_name: args.model_name = "google/siglip2-base-patch16-256"
    model, encode = darknet(args, device) if args.encoder == "darknet" else huggingface(args, device)
    started = time.time(); features = {}
    for representation in args.representations:
        chunks = []
        for start in range(0, len(rows), args.batch_size):
            images = [load_representation(args.dataset, row, representation) for row in rows[start:start + args.batch_size]]
            chunks.append(encode(images).half())
            if start and start % (args.batch_size * 100) == 0: print(representation, start, len(rows), flush=True)
        features[representation] = torch.cat(chunks)
    base = torch.load(args.dataset / "handcrafted_features.pt", map_location="cpu", weights_only=False)
    payload = {"rows": rows, "features": features, "classes": base["classes"], "encoder": args.encoder,
               "model_name": args.model_name, "parameters": sum(p.numel() for p in model.parameters()),
               "seconds": time.time() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, args.output)


if __name__ == "__main__": main()
