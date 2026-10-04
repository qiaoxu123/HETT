#!/usr/bin/env python3
"""Partial (last-block) multi-task fine-tuning for the selected SigLIP encoders."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from multiagent.models.multi_attribute_head import MultiAttributeHead  # noqa: E402
from multiagent.visual_attributes.dataset import load_representation, read_manifest  # noqa: E402
from multiagent.visual_attributes.metrics import classification_metrics, grouped_consistency  # noqa: E402


class ImageRows(Dataset):
    def __init__(self, root, rows, representation): self.root, self.rows, self.representation = root, rows, representation
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return load_representation(self.root, self.rows[index], self.representation), index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-name', required=True); parser.add_argument('--representation', default='crop')
    parser.add_argument('--epochs', type=int, default=3); parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=2e-5); parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--unfreeze-blocks', type=int, default=1)
    parser.add_argument('--max-train-samples', type=int); parser.add_argument('--max-val-samples', type=int)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name); model.requires_grad_(False)
    for block in model.vision_model.encoder.layers[-args.unfreeze_blocks:]: block.requires_grad_(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    base = torch.load(args.dataset / 'handcrafted_features.pt', map_location='cpu', weights_only=False)
    rows = read_manifest(args.dataset / 'manifest.jsonl'); classes = base['classes']
    class_index = {name: {value: index for index, value in enumerate(values)} for name, values in classes.items()}
    head = MultiAttributeHead(model.config.vision_config.hidden_size, {k: len(v) for k, v in classes.items()}).to(device)

    def collate(batch):
        images, indices = zip(*batch); pixels = processor(images=list(images), return_tensors='pt')['pixel_values']
        targets, masks = {}, {}
        for name, mapping in class_index.items():
            values = [rows[i]['labels'].get(name) for i in indices]
            masks[name] = torch.tensor([value in mapping for value in values])
            targets[name] = torch.tensor([mapping.get(value, 0) for value in values])
        return pixels, targets, masks, torch.tensor(indices)

    train_indices = [i for i, row in enumerate(rows) if row['split'] == 'train_seen']
    if args.max_train_samples: train_indices = train_indices[:args.max_train_samples]
    # Dataset-local indices need remapping to the full manifest for labels.
    train_rows = [rows[i] for i in train_indices]
    def train_collate(batch):
        images, local_indices = zip(*batch); full = [train_indices[i] for i in local_indices]
        pixels = processor(images=list(images), return_tensors='pt')['pixel_values']
        targets, masks = {}, {}
        for name, mapping in class_index.items():
            values = [rows[i]['labels'].get(name) for i in full]
            masks[name] = torch.tensor([value in mapping for value in values])
            targets[name] = torch.tensor([mapping.get(value, 0) for value in values])
        return pixels, targets, masks
    loader = DataLoader(ImageRows(args.dataset, train_rows, args.representation), batch_size=args.batch_size,
                        shuffle=True, num_workers=4, collate_fn=train_collate,
                        generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW([p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad], lr=args.learning_rate)
    started = time.time(); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); head.train(); losses = []
        for pixels, targets, masks in loader:
            optimizer.zero_grad(set_to_none=True); pixels = pixels.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}; masks = {k: v.to(device) for k, v in masks.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                features = model.get_image_features(pixel_values=pixels); outputs = head(features)
                loss = head.masked_loss(outputs, targets, masks)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses))}); print(history[-1], flush=True)

    model.eval(); head.eval(); result = {
        'config': vars(args) | {'dataset': str(args.dataset), 'output': str(args.output)}, 'history': history,
        'parameters': sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()),
        'trainable_parameters': sum(p.numel() for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad),
        'train_seconds': time.time() - started, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
        'splits': {},
    }
    for split in ('val_seen', 'val_unseen'):
        indices = [i for i, row in enumerate(rows) if row['split'] == split]
        if args.max_val_samples: indices = indices[:args.max_val_samples]
        eval_rows = [rows[i] for i in indices]; probabilities = {name: [] for name in classes}
        eval_loader = DataLoader(ImageRows(args.dataset, eval_rows, args.representation), batch_size=args.batch_size,
                                 num_workers=4, collate_fn=lambda batch: (processor(images=[x[0] for x in batch], return_tensors='pt')['pixel_values'], [x[1] for x in batch]))
        with torch.inference_mode():
            for pixels, _ in eval_loader:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    output = head(model.get_image_features(pixel_values=pixels.to(device)))
                for name, logits in output.items(): probabilities[name].append(torch.softmax(logits.float(), -1).cpu())
        split_result = {}
        for name, chunks in probabilities.items():
            probs = torch.cat(chunks).numpy(); mapping = class_index[name]
            selected = np.asarray([row['labels'].get(name) in mapping for row in eval_rows])
            targets = np.asarray([mapping.get(row['labels'].get(name), 0) for row in eval_rows])[selected]
            metrics = classification_metrics(probs[selected], targets)
            metrics['cross_height'] = grouped_consistency(probs[selected], [row['object_key'] for row, keep in zip(eval_rows, selected) if keep], [row['height_m'] for row, keep in zip(eval_rows, selected) if keep])
            split_result[name] = metrics
        result['splits'][split] = split_result
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
    torch.save({'model': model.state_dict(), 'head': head.state_dict(), 'classes': classes}, args.output / 'checkpoint.pt')


if __name__ == '__main__': main()
