#!/usr/bin/env python3
"""Controlled language-to-geometry probe with mirrored candidate positions."""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_candidate_geometry import build_split, mask_landmark_span


RELATION_PATTERNS = {
    "left": re.compile(r"(?:to|on) the left (?:side )?of (?:the )?masktoken"),
    "right": re.compile(r"(?:to|on) the right (?:side )?of (?:the )?masktoken"),
    "above": re.compile(r"(?:above|north of|north side of) (?:the )?masktoken"),
    "below": re.compile(r"(?:below|south of|south side of|under) (?:the )?masktoken"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path,
                        default=Path("runs/spatial_relation_probe/features.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/spatial_relation_probe/results.json"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--minimum-axis-m", type=float, default=10.0)
    return parser.parse_args()


def cue_names(masked_prompt):
    normalized = masked_prompt.lower().replace("[mask]", "masktoken")
    return [name for name, pattern in RELATION_PATTERNS.items()
            if pattern.search(normalized)]


def canonical_prompt(delta):
    horizontal = "right" if delta[0] > 0 else "left"
    vertical = "below" if delta[1] > 0 else "above"
    return f"The target is {horizontal} and {vertical} of [MASK]."


def make_candidates(delta):
    """True, horizontal mirror, vertical mirror, and 180-degree mirror."""
    dx, dy = float(delta[0]), float(delta[1])
    values = torch.tensor([[dx, dy], [-dx, dy], [dx, -dy], [-dx, -dy]])
    distance = torch.linalg.vector_norm(values, dim=-1).clamp_min(1e-6)
    return torch.cat((
        values,
        (torch.log1p(distance * 410.0) / math.log1p(410.0)).unsqueeze(-1),
        (values[:, 1] / distance).unsqueeze(-1),
        (values[:, 0] / distance).unsqueeze(-1),
    ), dim=-1)


def build_examples(data_root, split, minimum_axis_m):
    objects = json.load(open(data_root / "cityrefer/objects.json"))
    processed = json.load(open(data_root / "cityrefer/processed_descriptions.json"))
    trajectories = json.load(open(
        data_root / f"processed_citynav/citynav_{split}.json"))
    rows = build_split(trajectories, objects, processed, 9)
    examples, consistency = [], {name: [0, 0] for name in RELATION_PATTERNS}
    for row in rows:
        for query, name, center in zip(
                row["relation_queries"], row["names"], row["centers"]):
            prompt, found = mask_landmark_span(row["description"], query, name)
            if not found:
                continue
            cues = cue_names(prompt)
            if len(cues) != 1:
                continue
            delta = row["target"] - center
            if min(abs(delta[0]), abs(delta[1])) * 410.0 < minimum_axis_m:
                continue
            cue = cues[0]
            agrees = {
                "left": delta[0] < 0,
                "right": delta[0] > 0,
                "above": delta[1] < 0,
                "below": delta[1] > 0,
            }[cue]
            consistency[cue][0] += int(agrees)
            consistency[cue][1] += 1
            examples.append({
                "map_name": row["map_name"],
                "prompt": prompt,
                "synthetic_prompt": canonical_prompt(delta),
                "geometry": make_candidates(delta),
                "cue": cue,
            })
    return examples, consistency


@torch.inference_mode()
def encode_prompts(prompts, device, batch_size=256):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    model = AutoModel.from_pretrained(
        "bert-base-uncased", local_files_only=True).to(device).eval()
    unique = sorted(set(prompts))
    encoded = {}
    for offset in range(0, len(unique), batch_size):
        values = unique[offset:offset + batch_size]
        tokens = tokenizer(values, padding=True, truncation=True, max_length=128,
                           return_tensors="pt").to(device)
        hidden = model(**tokens).last_hidden_state
        is_mask = tokens.input_ids.eq(tokenizer.mask_token_id)
        mask_index = is_mask.float().argmax(-1)
        rows = torch.arange(len(values), device=device)
        vectors = hidden[rows, mask_index].float().cpu()
        encoded.update(zip(values, vectors))
    del model
    torch.cuda.empty_cache()
    return encoded


def tensorize(examples, encoded, field):
    return {
        "text": torch.stack([encoded[row[field]] for row in examples]).half(),
        "geometry": torch.stack([row["geometry"] for row in examples]).half(),
        "maps": [row["map_name"] for row in examples],
    }


class RelationGeometryProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = nn.Sequential(nn.Linear(768, 128), nn.LayerNorm(128), nn.SiLU(),
                                  nn.Linear(128, 64))
        self.geometry = nn.Sequential(nn.Linear(5, 64), nn.LayerNorm(64), nn.SiLU(),
                                      nn.Linear(64, 64))

    def forward(self, text, geometry):
        query = F.normalize(self.text(text.float()), dim=-1)
        key = F.normalize(self.geometry(geometry.float()), dim=-1)
        return torch.einsum("bkh,bh->bk", key, query) * 10.0


def shuffled_within_map(text, maps):
    result = torch.empty_like(text)
    groups = {}
    for index, map_name in enumerate(maps):
        groups.setdefault(map_name, []).append(index)
    for indices in groups.values():
        for offset, index in enumerate(indices):
            result[index] = text[indices[(offset + 1) % len(indices)]]
    return result


@torch.inference_mode()
def evaluate(model, data, device):
    logits = model(data["text"].to(device), data["geometry"].to(device)).cpu()
    shuffled = shuffled_within_map(data["text"], data["maps"])
    wrong_logits = model(shuffled.to(device), data["geometry"].to(device)).cpu()
    return {
        "count": len(logits),
        "four_way_accuracy": logits.argmax(-1).eq(0).float().mean().mul(100).item(),
        "same_map_wrong_text_accuracy": wrong_logits.argmax(-1).eq(0).float().mean().mul(100).item(),
        "true_vs_wrong_target_margin": (
            logits[:, 0] - logits[:, 1:].amax(-1)).mean().item(),
    }


def train_one(train, val, seed, epochs, device):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = RelationGeometryProbe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    dataset = TensorDataset(train["text"].float(), train["geometry"].float())
    best_state, best_accuracy = None, -1.0
    for _ in range(epochs):
        model.train()
        for text, geometry in DataLoader(dataset, batch_size=256, shuffle=True):
            logits = model(text.to(device), geometry.to(device))
            loss = F.cross_entropy(
                logits, torch.zeros(len(logits), dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        accuracy = evaluate(model, val, device)["four_way_accuracy"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


def summarize(runs, key):
    values = [run[key] for run in runs]
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cache.exists():
        cached = torch.load(args.cache, map_location="cpu", weights_only=False)
    else:
        examples, consistency = {}, {}
        prompts = []
        for split in ("train_seen", "val_seen", "val_unseen"):
            examples[split], consistency[split] = build_examples(
                args.data_root, split, args.minimum_axis_m)
            for row in examples[split]:
                prompts.extend((row["prompt"], row["synthetic_prompt"]))
            print(f"{split}: {len(examples[split])} controlled relations", flush=True)
        encoded = encode_prompts(prompts, device)
        cached = {"consistency": consistency, "real": {}, "synthetic": {}}
        for split, rows in examples.items():
            cached["real"][split] = tensorize(rows, encoded, "prompt")
            cached["synthetic"][split] = tensorize(rows, encoded, "synthetic_prompt")
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cached, args.cache)

    seeds = [int(value) for value in args.seeds.split(",") if value]
    result = {
        "config": {"minimum_axis_m": args.minimum_axis_m, "seeds": seeds,
                   "chance_four_way_accuracy": 25.0,
                   "split_rule": "train_seen -> val_seen selection -> val_unseen test"},
        "coordinate_consistency": cached["consistency"],
        "runs": {"synthetic": [], "real": []},
    }
    for mode in ("synthetic", "real"):
        for seed in seeds:
            model = train_one(cached[mode]["train_seen"], cached[mode]["val_seen"],
                              seed, args.epochs, device)
            result["runs"][mode].append({
                "seed": seed,
                "val_seen": evaluate(model, cached[mode]["val_seen"], device),
                "val_unseen": evaluate(model, cached[mode]["val_unseen"], device),
            })
            del model
            torch.cuda.empty_cache()
    result["summary"] = {}
    for mode, runs in result["runs"].items():
        unseen = [run["val_unseen"] for run in runs]
        result["summary"][mode] = {
            key: summarize(unseen, key) for key in (
                "four_way_accuracy", "same_map_wrong_text_accuracy",
                "true_vs_wrong_target_margin")
        }
    result["anti_leakage"] = {
        "probe_inputs": "masked relation text and four symmetric geometry candidates",
        "target_coordinates_used_for": "constructing supervised mirrored candidates only",
        "absolute_position_available_to_model": False,
        "candidate_index_feature_available_to_model": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
