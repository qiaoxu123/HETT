#!/usr/bin/env python3
"""Stage-1 training for the unified language/multi-landmark/BEV belief model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.belief_transformer import (
    BeliefTransformer, belief_loss, contrastive_input_loss, grid_centers,
    rasterize_landmarks, target_mask,
)
from scripts.train_relation_geometry_ranker import (
    opposite_direction_cues, structured_relation_cues,
)


DEFAULT_SOURCE = Path(
    "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/runs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/belief_transformer_stage1"))
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--max-landmarks", type=int, default=9)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overfit-count", type=int, default=0)
    parser.add_argument("--overfit-epochs", type=int, default=120)
    parser.add_argument("--bind-loss-weight", type=float, default=0.1)
    parser.add_argument("--relation-loss-weight", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=Path,
                        help="Evaluate a saved checkpoint without training")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def same_map_wrong_landmarks(rows):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["map_name"], []).append(index)
    result = []
    for index, row in enumerate(rows):
        group = groups[row["map_name"]]
        position = group.index(index)
        replacement = index
        for offset in range(1, len(group) + 1):
            candidate = group[(position + offset) % len(group)]
            if rows[candidate].get("landmark_names") != row.get("landmark_names"):
                replacement = candidate
                break
        result.append(replacement)
    return torch.tensor(result, dtype=torch.long)


class BeliefDataset(Dataset):
    """Materialize only model inputs plus separate target-derived labels."""

    def __init__(self, rows, relation_features, grid_size=20, max_landmarks=9):
        if relation_features.shape[:2] != (len(rows), max_landmarks):
            raise ValueError("relation feature cache does not align with rows")
        self.rows = rows
        self.relation = relation_features
        self.cues = structured_relation_cues(rows, max_landmarks)
        self.opposite_cues = opposite_direction_cues(self.cues)
        self.centers = torch.zeros(len(rows), max_landmarks, 2)
        self.boundaries = torch.zeros(len(rows), max_landmarks, 16, 2)
        self.valid = torch.zeros(len(rows), max_landmarks, dtype=torch.bool)
        for row_index, row in enumerate(rows):
            count = min(len(row.get("landmark_centers", [])),
                        len(row.get("landmark_boundaries", [])), max_landmarks)
            if count:
                self.centers[row_index, :count] = torch.tensor(
                    row["landmark_centers"][:count])
                self.boundaries[row_index, :count] = torch.tensor(
                    row["landmark_boundaries"][:count])
                self.valid[row_index, :count] = True
        self.instance_masks = rasterize_landmarks(
            self.boundaries, self.valid, grid_size).cpu()
        union = self.instance_masks.any(1).float()
        # Named-landmark occupancy and a thin boundary/edge channel.
        pooled = torch.nn.functional.max_pool2d(union[:, None], 3, 1, 1)[:, 0]
        edge = (pooled - union).clamp_min(0)
        self.map_bev = torch.stack((union, edge), dim=1)
        targets = torch.tensor([row["target"] for row in rows])
        self.labels = target_mask(targets, grid_size).cpu()
        self.targets = targets
        self.wrong_index = same_map_wrong_landmarks(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        wrong = int(self.wrong_index[index])
        return {
            "relation": self.relation[index], "cue": self.cues[index],
            "opposite_cue": self.opposite_cues[index],
            "centers": self.centers[index], "boundaries": self.boundaries[index],
            "valid": self.valid[index], "instance_masks": self.instance_masks[index],
            "map_bev": self.map_bev[index], "labels": self.labels[index],
            "target": self.targets[index],
            "wrong_centers": self.centers[wrong],
            "wrong_boundaries": self.boundaries[wrong],
            "wrong_valid": self.valid[wrong],
            "wrong_instance_masks": self.instance_masks[wrong],
            "wrong_map_bev": self.map_bev[wrong],
        }


MODEL_KEYS = ("relation", "cue", "centers", "boundaries", "valid",
              "instance_masks", "map_bev")


def to_model_args(batch, device, prefix="", cue_key=None):
    cue = batch[cue_key or "cue"].to(device)
    return (
        batch["relation"].to(device), cue,
        batch[prefix + "centers"].to(device),
        batch[prefix + "boundaries"].to(device),
        batch[prefix + "valid"].to(device),
        batch[prefix + "instance_masks"].to(device),
        batch[prefix + "map_bev"].to(device),
    )


def predict_metrics(logits, targets, grid_size):
    cells = grid_centers(grid_size).reshape(-1, 2)
    indices = logits.flatten(1).topk(5, dim=-1).indices.cpu()
    distance = torch.linalg.vector_norm(cells[indices] - targets[:, None], dim=-1) * 410.0
    return {
        "count": len(targets),
        "hit20": distance[:, 0].le(20).float().mean().mul(100).item(),
        "top5_hit20": distance.amin(-1).le(20).float().mean().mul(100).item(),
        "mean_ne": distance[:, 0].mean().item(),
    }


def select_landmark_representation(batch, mode):
    if mode == "normal":
        return batch
    result = dict(batch)
    valid = batch["valid"]
    if mode == "no_language":
        result["relation"] = torch.zeros_like(batch["relation"])
        result["cue"] = torch.zeros_like(batch["cue"])
        return result
    if mode == "no_structured_cue":
        result["cue"] = torch.zeros_like(batch["cue"])
        return result
    if mode == "first_landmark":
        result["valid"] = torch.zeros_like(valid)
        result["valid"][:, 0] = valid.any(-1)
        return result
    if mode != "average_landmarks":
        raise ValueError(mode)
    weight = valid[..., None].float()
    count = weight.sum(1).clamp_min(1.0)
    result["relation"] = torch.zeros_like(batch["relation"])
    result["relation"][:, 0] = (batch["relation"].float() * weight).sum(1) / count
    result["cue"] = torch.zeros_like(batch["cue"])
    result["cue"][:, 0] = batch["cue"].amax(1)
    result["centers"] = torch.zeros_like(batch["centers"])
    result["centers"][:, 0] = (batch["centers"] * weight).sum(1) / count
    boundary_weight = weight[..., None]
    result["boundaries"] = torch.zeros_like(batch["boundaries"])
    result["boundaries"][:, 0] = (
        batch["boundaries"] * boundary_weight).sum(1) / count[..., None]
    result["instance_masks"] = torch.zeros_like(batch["instance_masks"])
    result["instance_masks"][:, 0] = batch["instance_masks"].any(1)
    result["valid"] = torch.zeros_like(valid)
    result["valid"][:, 0] = valid.any(-1)
    return result


def grouped_metrics(logits, targets, landmark_count, grid_size):
    groups = {
        "one": landmark_count.eq(1),
        "two": landmark_count.eq(2),
        "three_plus": landmark_count.ge(3),
    }
    return {name: predict_metrics(logits[selected], targets[selected], grid_size)
            for name, selected in groups.items() if selected.any()}


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size):
    model.eval()
    normal, flipped, wrong, shuffled, no_language, no_cue = [], [], [], [], [], []
    first, averaged, targets, counts = [], [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        normal.append(model(*to_model_args(batch, device)).cpu())
        flipped.append(model(*to_model_args(
            batch, device, cue_key="opposite_cue")).cpu())
        wrong.append(model(*to_model_args(batch, device, prefix="wrong_")).cpu())
        order = torch.randperm(batch["valid"].shape[1])
        shuffled_batch = dict(batch)
        for key in ("relation", "cue", "centers", "boundaries", "valid",
                    "instance_masks"):
            shuffled_batch[key] = batch[key][:, order]
        shuffled.append(model(*to_model_args(shuffled_batch, device)).cpu())
        no_language.append(model(*to_model_args(
            select_landmark_representation(batch, "no_language"), device)).cpu())
        no_cue.append(model(*to_model_args(
            select_landmark_representation(batch, "no_structured_cue"), device)).cpu())
        first.append(model(*to_model_args(
            select_landmark_representation(batch, "first_landmark"), device)).cpu())
        averaged.append(model(*to_model_args(
            select_landmark_representation(batch, "average_landmarks"), device)).cpu())
        targets.append(batch["target"])
        counts.append(batch["valid"].sum(-1))
    normal, flipped, wrong, shuffled, no_language, no_cue, first, averaged = map(
        torch.cat, (normal, flipped, wrong, shuffled, no_language, no_cue,
                    first, averaged))
    targets = torch.cat(targets)
    counts = torch.cat(counts)
    result = {
        "normal": predict_metrics(normal, targets, model.grid_size),
        "opposite_relation": predict_metrics(flipped, targets, model.grid_size),
        "wrong_landmark": predict_metrics(wrong, targets, model.grid_size),
        "shuffled_landmark_order": predict_metrics(
            shuffled, targets, model.grid_size),
        "no_language": predict_metrics(no_language, targets, model.grid_size),
        "no_structured_cue": predict_metrics(no_cue, targets, model.grid_size),
        "first_landmark_only": predict_metrics(first, targets, model.grid_size),
        "averaged_landmarks": predict_metrics(averaged, targets, model.grid_size),
        "flip_changes_top1_percent": (normal.flatten(1).argmax(-1) !=
                                      flipped.flatten(1).argmax(-1)).float().mean().mul(100).item(),
        "shuffle_max_logit_difference": (normal - shuffled).abs().max().item(),
    }
    result["normal"]["by_landmark_count"] = grouped_metrics(
        normal, targets, counts, model.grid_size)
    result["averaged_landmarks"]["by_landmark_count"] = grouped_metrics(
        averaged, targets, counts, model.grid_size)
    result["first_landmark_only"]["by_landmark_count"] = grouped_metrics(
        first, targets, counts, model.grid_size)
    return result


def train(model, dataset, device, epochs, batch_size, learning_rate,
          bind_weight, relation_weight, validation=None):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    best_state, best_key, best_epoch = None, None, 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            labels = batch["labels"].to(device)
            normal = model(*to_model_args(batch, device))
            loss = belief_loss(normal, labels)
            if bind_weight:
                wrong = model(*to_model_args(batch, device, prefix="wrong_"))
                loss = loss + bind_weight * contrastive_input_loss(normal, wrong, labels)
            if relation_weight:
                flipped = model(*to_model_args(
                    batch, device, cue_key="opposite_cue"))
                loss = loss + relation_weight * contrastive_input_loss(
                    normal, flipped, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(labels)
        if (validation is not None or epoch == 0
                or (epoch + 1) % max(1, epochs // 10) == 0):
            selected_data = validation if validation is not None else dataset
            metric = evaluate(model, selected_data, device, batch_size)["normal"]
            print(f"epoch={epoch + 1}/{epochs} loss={total / len(dataset):.4f} "
                  f"hit20={metric['hit20']:.2f}", flush=True)
            key = metric["hit20"], -metric["mean_ne"]
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch + 1
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), best_epoch


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = args.candidate_dir / "train_seen.jsonl"
    val_path = args.candidate_dir / "val_seen.jsonl"
    relation = torch.load(args.relation_cache, map_location="cpu", weights_only=True)
    train_rows = load_rows(train_path)
    val_rows = load_rows(val_path)
    train_features = relation["train_seen"]
    if args.overfit_count:
        train_rows = train_rows[:args.overfit_count]
        train_features = train_features[:args.overfit_count]
    train_data = BeliefDataset(
        train_rows, train_features, args.grid_size, args.max_landmarks)
    val_data = BeliefDataset(
        val_rows, relation["val_seen"], args.grid_size, args.max_landmarks)
    model = BeliefTransformer(
        grid_size=args.grid_size, hidden=args.hidden, layers=args.layers)
    epochs = args.overfit_epochs if args.overfit_count else args.epochs
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        model.to(device).eval()
        best_epoch = saved.get("config", {}).get("best_epoch", 0)
        mode = "evaluation_only"
    else:
        model, best_epoch = train(
            model, train_data, device, epochs, args.batch_size,
            args.learning_rate, args.bind_loss_weight, args.relation_loss_weight,
            validation=None if args.overfit_count else val_data)
        mode = "overfit" if args.overfit_count else "train_seen"
    result = {
        "stage": "language + independent landmark instances + named-landmark BEV",
        "mode": mode,
        "train_count": len(train_data), "validation_count": len(val_data),
        "epochs": epochs, "best_epoch": best_epoch,
        "train": evaluate(model, train_data, device, args.batch_size),
        "val_seen": evaluate(model, val_data, device, args.batch_size),
        "anti_leakage": {
            "training_split": "train_seen",
            "checkpoint_or_hyperparameter_split": "val_seen",
            "val_unseen_loaded": False,
            "target_coordinates_in_model_forward": False,
            "target_coordinates_usage": "target mask labels and final metrics only",
        },
        "data": {
            "train_path": str(train_path.resolve()),
            "val_path": str(val_path.resolve()),
            "relation_cache": str(args.relation_cache.resolve()),
            "train_sha256": file_sha256(train_path),
            "val_sha256": file_sha256(val_path),
            "relation_sha256": file_sha256(args.relation_cache),
        },
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("evaluation" if args.checkpoint else
              f"overfit{args.overfit_count}" if args.overfit_count else "train3epoch")
    (args.output_dir / f"results_{suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n")
    if not args.checkpoint:
        checkpoint_config = dict(result["config"])
        checkpoint_config["best_epoch"] = best_epoch
        torch.save({"model": model.state_dict(), "config": checkpoint_config},
                   args.output_dir / f"model_{suffix}.pt")
    print(json.dumps(result["train"], indent=2), flush=True)
    print("saved", args.output_dir / f"results_{suffix}.json", flush=True)


if __name__ == "__main__":
    main()
