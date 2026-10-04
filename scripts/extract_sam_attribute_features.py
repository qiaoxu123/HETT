#!/usr/bin/env python3
"""Sampled SAM box-prompt segmentation benchmark with SigLIP2 features."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from multiagent.visual_attributes.dataset import load_representation, read_manifest  # noqa: E402


def select_rows(rows, limits, seed):
    random.seed(seed); result = []
    for split, limit in limits.items():
        keys = sorted({row['object_key'] for row in rows if row['split'] == split}); random.shuffle(keys)
        selected = set(keys[:limit]); result.extend(row for row in rows if row['split'] == split and row['object_key'] in selected)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sam-source', required=True, type=Path); parser.add_argument('--sam-checkpoint', required=True, type=Path)
    parser.add_argument('--model-name', default='google/siglip2-base-patch16-256')
    parser.add_argument('--train-objects', type=int, default=500); parser.add_argument('--val-objects', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=24); parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(); sys.path.insert(0, str(args.sam_source))
    from segment_anything import SamPredictor, sam_model_registry
    from transformers import AutoImageProcessor, AutoModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry['vit_b'](checkpoint=str(args.sam_checkpoint)).to(device).eval()
    predictor = SamPredictor(sam); processor = AutoImageProcessor.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name).to(device).eval().requires_grad_(False)
    all_rows = read_manifest(args.dataset / 'manifest.jsonl')
    rows = select_rows(all_rows, {'train_seen': args.train_objects, 'val_seen': args.val_objects, 'val_unseen': args.val_objects}, args.seed)
    chunks = {'sam_masked': [], 'crop': []}; pending = {'sam_masked': [], 'crop': []}
    ious, scores = [], []; started = time.time()
    def flush():
        if not pending['sam_masked']: return
        count = len(pending['sam_masked']); combined = pending['sam_masked'] + pending['crop']
        pixels = processor(images=combined, return_tensors='pt')['pixel_values'].to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            value = encoder.get_image_features(pixel_values=pixels)
        chunks['sam_masked'].append(value[:count].half().cpu()); chunks['crop'].append(value[count:].half().cpu())
        pending['sam_masked'].clear(); pending['crop'].clear()
    for index, row in enumerate(rows):
        image = load_representation(args.dataset, row, 'whole'); polygon = np.asarray(row['contour_uv'], np.int32)
        oracle = cv2.fillPoly(np.zeros(image.shape[:2], np.uint8), [polygon], 1).astype(bool)
        x, y, width, height = cv2.boundingRect(polygon)
        box = np.asarray([max(x, 0), max(y, 0), min(x + width, image.shape[1] - 1), min(y + height, image.shape[0] - 1)])
        predictor.set_image(image); masks, quality, _ = predictor.predict(box=box, multimask_output=True)
        best = int(np.argmax(quality)); mask = masks[best]
        intersection = np.logical_and(mask, oracle).sum(); union = np.logical_or(mask, oracle).sum()
        ious.append(float(intersection / max(union, 1))); scores.append(float(quality[best]))
        masked = np.full_like(image, 127); masked[mask] = image[mask]
        pending['sam_masked'].append(masked); pending['crop'].append(load_representation(args.dataset, row, 'crop'))
        if len(pending['sam_masked']) >= args.batch_size: flush()
        if index and index % 500 == 0: print(index, len(rows), float(np.mean(ious)), flush=True)
    flush(); base = torch.load(args.dataset / 'handcrafted_features.pt', map_location='cpu', weights_only=False)
    payload = {'rows': rows, 'features': {name: torch.cat(value) for name, value in chunks.items()}, 'classes': base['classes'],
               'encoder': 'siglip2_sam_vit_b', 'model_name': args.model_name}
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, args.output)
    metrics = {'samples': len(rows), 'objects': {'train_seen': args.train_objects, 'val_seen': args.val_objects, 'val_unseen': args.val_objects},
               'mask_iou_mean': float(np.mean(ious)), 'mask_iou_median': float(np.median(ious)),
               'sam_predicted_iou_mean': float(np.mean(scores)), 'seconds': time.time() - started,
               'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0}
    (args.output.parent / 'segmentation_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__': main()
