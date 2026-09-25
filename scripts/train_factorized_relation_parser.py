#!/usr/bin/env python3
"""Train a high-precision relation parser and keep geometry deterministic.

The parser is supervised only by explicit per-landmark direction phrases whose
training target agrees with the audited reference frame.  Navigation distance
labels never supervise the parser.  A confidence threshold and the final
visual/relation fusion weight are selected on val_seen, then frozen for a
single val_unseen diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.path_local_verifier import LandmarkAwareCandidateVerifier
from scripts.audit_spatial_relation_probe import cue_names
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE
from scripts.train_landmark_visual_verifier import predict
from scripts.train_path_local_candidate_verifier import rows_for_records
from scripts.train_relation_geometry_ranker import (
    audit_consistent_landmark_geometry,
    load_rows,
    structured_relation_cues,
)
from scripts.train_rgb_candidate_ranker import metrics
from scripts.validate_candidate_geometry import mask_landmark_span


DIRECTIONS = ("left", "right", "above", "below")
NONE_INDEX = len(DIRECTIONS)


class RelationParser(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(hidden, len(DIRECTIONS) + 1))

    def forward(self, relation):
        return self.net(relation.float())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--seen-features", type=Path, default=Path(
        "runs/landmark_visual_verifier/relative_margin_2048_fullval_features.pt"))
    parser.add_argument("--unseen-features", type=Path, default=Path(
        "runs/landmark_visual_verifier/frozen_val_unseen_features.pt"))
    parser.add_argument("--records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/expanded_train_records.pt"))
    parser.add_argument("--unseen-records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/val_unseen_records.pt"))
    parser.add_argument("--visual-checkpoint", type=Path, default=Path(
        "runs/landmark_visual_verifier/relative_margin_2048_fullval/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/factorized_relation_parser"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--minimum-axis-m", type=float, default=10.0)
    parser.add_argument("--minimum-val-precision", type=float, default=0.70)
    parser.add_argument("--max-landmarks", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def strict_cues(rows, max_landmarks):
    cues = torch.zeros(len(rows), max_landmarks, len(DIRECTIONS))
    lookup = {name: index for index, name in enumerate(DIRECTIONS)}
    for row_index, row in enumerate(rows):
        for landmark_index, (query, name) in enumerate(zip(
                row.get("relation_queries", [])[:max_landmarks],
                row.get("landmark_names", [])[:max_landmarks])):
            prompt, found = mask_landmark_span(row["description"], query, name)
            names = cue_names(prompt) if found else []
            if len(names) == 1:
                cues[row_index, landmark_index, lookup[names[0]]] = 1
    return cues


def target_geometry(rows, max_landmarks):
    target_rows = []
    for row in rows:
        target_rows.append({
            "candidates": [row["target"]],
            "landmark_centers": row.get("landmark_centers", []),
            "landmark_boundaries": row.get("landmark_boundaries", []),
        })
    geometry, valid = audit_consistent_landmark_geometry(
        target_rows, max_landmarks)
    return geometry[:, 0].float(), valid


def direction_projection(geometry):
    x, y = geometry[..., 0], geometry[..., 1]
    return torch.stack((-x, x, -y, y), dim=-1)


def parser_examples(rows, relation, max_landmarks, minimum_axis_m):
    strict = strict_cues(rows, max_landmarks)
    broad = structured_relation_cues(rows, max_landmarks)[..., :4]
    target, valid = target_geometry(rows, max_landmarks)
    projection = direction_projection(target)
    strict_count = strict.sum(-1)
    strict_index = strict.argmax(-1)
    strict_margin = projection.gather(-1, strict_index[..., None]).squeeze(-1)
    trusted_positive = (valid & strict_count.eq(1)
                        & strict_margin.gt(minimum_axis_m / 410.0))
    trusted_none = valid & broad.sum(-1).eq(0)
    selected = trusted_positive | trusted_none
    labels = torch.full(valid.shape, NONE_INDEX, dtype=torch.long)
    labels[trusted_positive] = strict_index[trusted_positive]
    return relation[selected].float(), labels[selected], {
        "trusted_direction": int(trusted_positive.sum()),
        "trusted_none": int(trusted_none.sum()),
    }


@torch.inference_mode()
def parser_logits(model, relation, device, batch_size=1024):
    output = []
    flat = relation.reshape(-1, relation.shape[-1])
    for offset in range(0, len(flat), batch_size):
        output.append(model(flat[offset:offset + batch_size].to(device)).cpu())
    return torch.cat(output).reshape(*relation.shape[:-1], -1)


def train_parser(train_x, train_y, val_x, val_y, args, device):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model = RelationParser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    counts = torch.bincount(train_y, minlength=len(DIRECTIONS) + 1).float()
    class_weight = counts.sum() / (
        (len(DIRECTIONS) + 1) * counts.clamp_min(1))
    # Do not let the abundant NONE class erase the sparse directional signal.
    class_weight = class_weight.to(device)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for relation, label in DataLoader(
                TensorDataset(train_x, train_y), batch_size=args.batch_size,
                shuffle=True):
            logits = model(relation.to(device))
            loss = nn.functional.cross_entropy(
                logits, label.to(device), weight=class_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = parser_logits(model, val_x, device)
            prediction = logits.argmax(-1)
            per_class = []
            for class_index in range(len(DIRECTIONS) + 1):
                selected = val_y.eq(class_index)
                if selected.any():
                    per_class.append(prediction[selected].eq(
                        val_y[selected]).float().mean())
            macro_accuracy = torch.stack(per_class).mean().item()
            direction_selected = val_y.lt(NONE_INDEX)
            direction_accuracy = prediction[direction_selected].eq(
                val_y[direction_selected]).float().mean().item()
            accuracy = prediction.eq(val_y).float().mean().item()
            loss = nn.functional.cross_entropy(
                logits, val_y, weight=class_weight.cpu()).item()
        key = macro_accuracy, direction_accuracy, -loss, -epoch
        if best is None or key > best[0]:
            best = key, copy.deepcopy(model.state_dict()), epoch
        print(f"epoch={epoch} val_accuracy={accuracy * 100:.2f} "
              f"macro={macro_accuracy * 100:.2f} "
              f"direction={direction_accuracy * 100:.2f} loss={loss:.4f}", flush=True)
    model.load_state_dict(best[1]); return model.eval(), best[2]


def select_threshold(logits, rows, valid, args):
    probability = logits.softmax(-1)
    confidence, prediction = probability[..., :4].max(-1)
    target, _ = target_geometry(rows, args.max_landmarks)
    projection = direction_projection(target)
    margin = projection.gather(-1, prediction[..., None]).squeeze(-1)
    measurable = valid & margin.abs().gt(args.minimum_axis_m / 410.0)
    best = None
    for step in range(50, 100, 5):
        threshold = step / 100.0
        selected = measurable & confidence.ge(threshold)
        if not selected.any():
            continue
        precision = margin[selected].gt(0).float().mean().item()
        coverage = selected.float().mean().item()
        passes = precision >= args.minimum_val_precision
        key = int(passes), coverage if passes else precision, precision
        if best is None or key > best[0]:
            best = key, threshold, precision, coverage
    return best[1:]


def inferred_cues(model, relation, rows, valid, threshold, args, device):
    logits = parser_logits(model, relation, device)
    probability = logits.softmax(-1)
    confidence, prediction = probability[..., :4].max(-1)
    strict = strict_cues(rows, args.max_landmarks)
    strict_valid = strict.sum(-1).eq(1)
    cues = torch.zeros_like(strict)
    inferred = valid & ~strict_valid & confidence.ge(threshold)
    cues.scatter_(-1, prediction[..., None], 1.0)
    cues *= inferred[..., None]
    cues[strict_valid] = strict[strict_valid]
    active = cues.sum(-1).eq(1) & valid
    return cues, active, confidence


def relation_score(rows, cues, active, args):
    geometry, _ = audit_consistent_landmark_geometry(rows, args.max_landmarks)
    projection = direction_projection(geometry.float())
    count = active.sum(-1).clamp_min(1).float()
    score = (projection * cues[:, None]).sum((2, 3)) / count[:, None]
    score[active.sum(-1).eq(0)] = 0
    flipped = cues[..., [1, 0, 3, 2]]
    flipped_score = (projection * flipped[:, None]).sum((2, 3)) / count[:, None]
    flipped_score[active.sum(-1).eq(0)] = 0
    return score, flipped_score


def load_records(path, split, count):
    records = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(records, dict):
        records = records[split]
    return records[:count]


def pair_accuracy(distance, score, selected):
    unique = distance.le(20).sum(-1).eq(1)
    mask = selected & unique
    if not mask.any():
        return {"count": 0, "accuracy": None}
    truth = distance.argmin(-1)
    return {"count": int(mask.sum()), "accuracy": float(
        score.argmax(-1)[mask].eq(truth[mask]).float().mean() * 100)}


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {split: load_rows(args.candidate_dir / f"{split}.jsonl", 5)
            for split in ("train_seen", "val_seen", "val_unseen")}
    relation = torch.load(args.relation_cache, map_location="cpu", weights_only=True)
    train_x, train_y, train_counts = parser_examples(
        rows["train_seen"], relation["train_seen"], args.max_landmarks,
        args.minimum_axis_m)
    val_x, val_y, val_counts = parser_examples(
        rows["val_seen"], relation["val_seen"], args.max_landmarks,
        args.minimum_axis_m)
    model, epoch = train_parser(train_x, train_y, val_x, val_y, args, device)
    val_logits = parser_logits(model, relation["val_seen"], device)
    _, val_valid = target_geometry(rows["val_seen"], args.max_landmarks)
    threshold, threshold_precision, threshold_coverage = select_threshold(
        val_logits, rows["val_seen"], val_valid, args)

    seen = torch.load(args.seen_features, map_location="cpu", weights_only=False)[
        "val_seen"]
    unseen = torch.load(args.unseen_features, map_location="cpu", weights_only=False)
    split_data = {"val_seen": seen, "val_unseen": unseen}
    split_records = {
        "val_seen": load_records(args.records, "val_seen", len(seen["distance"])),
        "val_unseen": load_records(
            args.unseen_records, "val_unseen", len(unseen["distance"])),
    }
    visual_state = torch.load(
        args.visual_checkpoint, map_location="cpu", weights_only=True)
    visual_model = LandmarkAwareCandidateVerifier(
        int(visual_state["vision_dim"]),
        relation_dim=int(visual_state.get("relation_dim", 768))).to(device)
    visual_model.load_state_dict(visual_state["model"]); visual_model.eval()

    prepared = {}
    for split, data in split_data.items():
        current_rows = rows_for_records(
            args.candidate_dir, split, split_records[split])
        valid = data["valid"].bool()
        cues, active, confidence = inferred_cues(
            model, data["relation"][..., :768], current_rows, valid,
            threshold, args, device)
        score, flipped = relation_score(current_rows, cues, active, args)
        prepared[split] = {
            "visual": predict(visual_model, data, device),
            "relation": score, "flipped": flipped,
            "active_episode": active.any(-1),
            "active_landmark_percent": float(
                active.sum() / valid.sum().clamp_min(1) * 100),
        }

    visual_scale = (prepared["val_seen"]["visual"][:, 0]
                    - prepared["val_seen"]["visual"][:, 1]).std()
    relation_scale = (prepared["val_seen"]["relation"][:, 0]
                      - prepared["val_seen"]["relation"][:, 1]).std().clamp_min(1e-6)
    best = None
    for step in range(41):
        alpha = step / 20.0
        fused = (prepared["val_seen"]["visual"] / visual_scale
                 + alpha * prepared["val_seen"]["relation"] / relation_scale)
        value = metrics(seen["distance"], fused.argmax(-1))
        key = value["hit20"], -value["mean_ne"], -alpha
        if best is None or key > best[0]:
            best = key, alpha
    alpha = best[1]

    evaluation = {}
    for split, data in split_data.items():
        item = prepared[split]
        fused = item["visual"] / visual_scale + alpha * item["relation"] / relation_scale
        flipped = item["visual"] / visual_scale + alpha * item["flipped"] / relation_scale
        evaluation[split] = {
            "visual_only": metrics(data["distance"], item["visual"].argmax(-1)),
            "factorized_relation_fusion": metrics(
                data["distance"], fused.argmax(-1)),
            "flipped_relation_control": metrics(
                data["distance"], flipped.argmax(-1)),
            "active_episode_percent": float(
                item["active_episode"].float().mean() * 100),
            "active_landmark_percent": item["active_landmark_percent"],
            "pair_accuracy": {
                "visual": pair_accuracy(
                    data["distance"], item["visual"], item["active_episode"]),
                "correct": pair_accuracy(
                    data["distance"], fused, item["active_episode"]),
                "flipped": pair_accuracy(
                    data["distance"], flipped, item["active_episode"]),
            },
        }
    result = {
        "experiment": "factorized high-confidence relation parser",
        "training_examples": train_counts,
        "validation_examples": val_counts,
        "selected_epoch": epoch,
        "selected_on_val_seen": {
            "confidence_threshold": threshold,
            "target_geometry_precision": threshold_precision,
            "measurable_landmark_coverage": threshold_coverage,
            "fusion_alpha": alpha,
            "visual_scale": float(visual_scale),
            "relation_scale": float(relation_scale),
        },
        **evaluation,
        "anti_leakage": {
            "parser_training": "train_seen only",
            "threshold_and_alpha_selection": "val_seen only",
            "target_in_parser_or_navigation_forward": False,
            "training_target_use": "filter ambiguous relation labels only",
            "test_unseen_loaded": False,
        },
    }
    torch.save({"model": model.state_dict(), "threshold": threshold,
                "alpha": alpha}, args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
