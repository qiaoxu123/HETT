#!/usr/bin/env python3
"""Train per-landmark, relation-conditioned signed RGB belief updates."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.belief_transformer import (
    BeliefTransformer, RelationConditionedTemporalBeliefTransformer,
    belief_loss, contrastive_input_loss,
)
from scripts.train_belief_transformer import predict_metrics, to_model_args
from scripts.train_temporal_belief_transformer import (
    aligned_dataset, build_feature_cache, evidence_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(
        "runs/temporal_belief_candidate_rgb_gate5_large"))
    parser.add_argument("--feature-cache", type=Path, default=Path(
        "runs/temporal_belief_candidate_rgb_gate5_large/semantic_bev_support20.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/"
        "runs/candidate_rgb_data/geometry_top5_oof_grouped_enriched"))
    parser.add_argument("--relation-cache", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/"
        "runs/relation_geometry_ranker_contrastive/relation_features.pt"))
    parser.add_argument("--semantic-checkpoint", type=Path, default=Path(
        "runs/pointcloud_semantic_mixed_gate2/model.pt"))
    parser.add_argument("--static-checkpoint", type=Path, default=Path(
        "runs/belief_transformer_stage1_relative/model_train3epoch.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/relation_conditioned_temporal_belief"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--temporal-hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--evidence-loss-weight", type=float, default=0.1)
    parser.add_argument("--control-loss-weight", type=float, default=0.3)
    parser.add_argument("--overfit-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    # Required by the shared, already-cached feature builder.
    parser.add_argument("--evidence-radius-m", type=float, default=20.0)
    return parser.parse_args()


def evidence_args(batch, device, mode="normal"):
    normal = to_model_args(batch, device)
    if mode == "normal":
        return normal
    if mode == "opposite_relation":
        values = list(normal)
        values[1] = batch["opposite_cue"].to(device)
        return tuple(values)
    if mode == "wrong_landmark":
        return (
            normal[0], normal[1], batch["wrong_centers"].to(device),
            batch["wrong_boundaries"].to(device),
            batch["wrong_valid"].to(device),
            batch["wrong_instance_masks"].to(device),
            batch["wrong_map_bev"].to(device),
        )
    raise ValueError(mode)


def frame_loss(values, labels):
    weights = torch.linspace(0.5, 1.0, values.shape[1], device=values.device)
    return sum(weight * belief_loss(values[:, index], labels)
               for index, weight in enumerate(weights)) / weights.sum()


def control_loss(correct, control, labels):
    return sum(contrastive_input_loss(
        correct[:, index], control[:, index], labels, margin=0.2)
        for index in range(correct.shape[1])) / correct.shape[1]


def training_loss(model, batch, args, device):
    static = to_model_args(batch, device)
    semantic = batch["semantic_bev"].to(device)
    visibility = batch["evidence_support"].to(device)
    labels = batch["labels"].to(device)
    normal = model(static, semantic, visibility)
    opposite = model(static, semantic, visibility, evidence_args(
        batch, device, "opposite_relation"))
    wrong = model(static, semantic, visibility, evidence_args(
        batch, device, "wrong_landmark"))
    return (
        frame_loss(normal["belief_logits"], labels)
        + args.evidence_loss_weight * evidence_loss(
            normal["raw_evidence"], labels, visibility)
        + args.control_loss_weight * (
            control_loss(normal["belief_logits"], opposite["belief_logits"], labels)
            + control_loss(normal["belief_logits"], wrong["belief_logits"], labels)))


@torch.inference_mode()
def evaluate(model, dataset, args, device):
    model.eval()
    names = ("normal", "opposite_relation", "wrong_landmark",
             "wrong_pose", "no_rgb")
    collected = {name: [] for name in names}
    initial, targets = [], []
    for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
        static = to_model_args(batch, device)
        semantic = batch["semantic_bev"].to(device)
        visibility = batch["evidence_support"].to(device)
        normal = model(static, semantic, visibility)
        outputs = {
            "normal": normal,
            "opposite_relation": model(
                static, semantic, visibility,
                evidence_args(batch, device, "opposite_relation")),
            "wrong_landmark": model(
                static, semantic, visibility,
                evidence_args(batch, device, "wrong_landmark")),
            # Wrong RGB with the original pose/support isolates visual geometry.
            "wrong_pose": model(
                static, batch["shuffled_semantic_bev"].to(device), visibility),
            "no_rgb": model(static, torch.zeros_like(semantic), visibility),
        }
        initial.append(normal["initial_logits"].cpu())
        targets.append(batch["target"])
        for name, output in outputs.items():
            collected[name].append(output["belief_logits"].cpu())
    initial = torch.cat(initial); targets = torch.cat(targets)
    collected = {name: torch.cat(value) for name, value in collected.items()}

    def metrics_by_frame(value):
        return [predict_metrics(value[:, step], targets, args.grid_size)
                for step in range(value.shape[1])]

    result = {
        "static_initial": predict_metrics(initial, targets, args.grid_size),
        **{name: metrics_by_frame(value) for name, value in collected.items()},
    }
    best_step = max(range(collected["normal"].shape[1]), key=lambda step: (
        result["normal"][step]["hit20"],
        -result["normal"][step]["mean_ne"]))
    correct = result["normal"][best_step]
    result["selected_frame"] = best_step + 1
    result["control_gap_pp"] = {
        name: correct["hit20"] - result[name][best_step]["hit20"]
        for name in names[1:]
    }
    result["normal_changes_top1_percent"] = float((
        collected["normal"][:, best_step].flatten(1).argmax(-1)
        != initial.flatten(1).argmax(-1)).float().mean() * 100)
    return result


def train(model, train_data, val_data, args, device):
    for parameter in model.static_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-3)
    best_key, best_state, best_epoch, history = None, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); model.static_model.eval(); losses = []
        for batch in DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True):
            loss = training_loss(model, batch, args, device)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss.detach()))
        value = evaluate(model, val_data, args, device)
        step = value["selected_frame"] - 1
        normal = value["normal"][step]
        minimum_control_gap = min(
            value["control_gap_pp"]["opposite_relation"],
            value["control_gap_pp"]["wrong_landmark"])
        key = (normal["hit20"], minimum_control_gap, -normal["mean_ne"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "validation": value})
        print(f"epoch={epoch}/{args.epochs} loss={np.mean(losses):.4f} "
              f"hit20={normal['hit20']:.2f} frame={step + 1} "
              f"relation_gap={value['control_gap_pp']['opposite_relation']:.2f} "
              f"landmark_gap={value['control_gap_pp']['wrong_landmark']:.2f}",
              flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), best_epoch, history


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = build_feature_cache(args, device)
    relation = torch.load(args.relation_cache, map_location="cpu", weights_only=True)
    train_data = aligned_dataset(
        args, "train_seen", features["splits"]["train_seen"], relation)
    val_data = aligned_dataset(
        args, "val_seen", features["splits"]["val_seen"], relation)
    if args.overfit_count:
        count = min(args.overfit_count, len(train_data))
        train_data = Subset(train_data, range(count))
        val_data = train_data
    static = BeliefTransformer(
        grid_size=args.grid_size, hidden=args.hidden, layers=args.layers)
    saved = torch.load(args.static_checkpoint, map_location="cpu", weights_only=True)
    static.load_state_dict(saved["model"])
    model = RelationConditionedTemporalBeliefTransformer(
        static, hidden=args.temporal_hidden).to(device)
    model, epoch, history = train(model, train_data, val_data, args, device)
    result = {
        "stage": "per-named-landmark relation-conditioned signed RGB update",
        "selected_epoch_on_val_seen": epoch,
        "train_seen": evaluate(model, train_data, args, device),
        "val_seen": evaluate(model, val_data, args, device),
        "history": history,
        "gate": {
            "requirement": (
                "normal RGB must improve over the frozen static prior and beat "
                "opposite-relation, wrong-landmark, wrong-pose and no-RGB controls"),
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
        },
        "anti_leakage": {
            "model_forward_target_coordinate": False,
            "frame_source": "predicted-candidate views fixed before training",
            "target_usage": "training labels and final metrics only",
            "control_isolation": (
                "controls preserve the correct static prior and alter only the "
                "online evidence condition"),
        },
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "selected_epoch": epoch,
                "config": result["config"]}, args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["val_seen"], indent=2))


if __name__ == "__main__":
    main()
