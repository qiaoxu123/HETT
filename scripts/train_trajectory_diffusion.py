#!/usr/bin/env python3
"""Train the first target-free multi-trajectory diffusion prototype.

The initial gate intentionally overfits a small ``train_seen`` subset.  Goal
coordinates are never model inputs: the supervision paths are generated only
from frozen belief modes and the current pose.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import OfficialProtocol, belief_distinct_topk_xy
from independent_nav.trajectory_diffusion import (
    ConditionalUnet1D, bounded_endpoint_projection, ddim_sample,
    make_ddpm_scheduler,
)
from independent_nav.trajectory_library import (
    generate_rule_trajectory_set, stack_trajectory_set,
)
from scripts.build_temporal_belief_dataset import pose_xy_yaw
from scripts.evaluate_official_static_belief import (
    DEFAULT_SOURCE, BeliefInferenceDataset, load_rows, predict_beliefs,
)
from scripts.train_trajectory_hypothesis_scorer import normalized_start


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--belief-checkpoint", type=Path, default=Path(
        "runs/belief_transformer_stage1_relative/model_train3epoch.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/trajectory_diffusion_smoke"))
    parser.add_argument("--max-train", type=int, default=32)
    parser.add_argument("--holdout", type=int, default=0,
                        help="Disjoint train_seen episodes immediately after training slice")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Continue from a previously saved model state")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def standardize_belief(value):
    flat = value.flatten(1)
    return ((value - flat.mean(1)[:, None, None]) /
            flat.std(1).clamp_min(1e-5)[:, None, None])


def build_data(args, device, start=0, count=None):
    stop = None if count is None or count <= 0 else start + count
    rows = load_rows(args.candidate_dir / "train_seen.jsonl")[start:stop]
    trajectories = json.loads((args.data_root / "processed_citynav" /
                               "citynav_train_seen.json").read_text())[start:stop]
    relation = torch.load(
        args.relation_cache, map_location="cpu", weights_only=True
    )["train_seen"][start:stop]
    inference = BeliefInferenceDataset(rows, relation)
    belief, _ = predict_beliefs(
        args.belief_checkpoint, inference, device, args.batch_size)
    belief = standardize_belief(belief)
    starts, pose_budget, libraries = [], [], []
    for index, (row, trajectory, belief_map) in enumerate(
            zip(rows, trajectories, belief)):
        map_name = f'{trajectory["area"]}_block_{trajectory["block"]}'
        if map_name != row["map_name"] or trajectory["descriptions"][0] != row["description"]:
            raise ValueError(f"row alignment mismatch at train_seen:{index}")
        start = normalized_start(trajectory["trajectory"][0], map_name)
        _, _, yaw = pose_xy_yaw(trajectory["trajectory"][0])
        modes = belief_distinct_topk_xy(belief_map, 5, 25.0)
        library = stack_trajectory_set(
            generate_rule_trajectory_set(start, yaw, modes))
        starts.append(start)
        pose_budget.append((start[0], start[1], math.sin(yaw), math.cos(yaw), 1.0))
        libraries.append(library)
    return {
        "belief": belief,
        "relation": inference.relation,
        "cue": inference.cues,
        "centers": inference.centers,
        "valid": inference.valid,
        "pose_budget": torch.tensor(pose_budget, dtype=torch.float32),
        "actions": torch.stack([value["actions"] for value in libraries]),
        "action_valid": torch.stack([value["valid"] for value in libraries]),
        "length": torch.stack([value["movement_steps"] for value in libraries]),
        "rule_positions": torch.stack([value["positions"] for value in libraries]),
        "start": torch.tensor(starts, dtype=torch.float32),
        # Metric-only labels. These are never returned by VariantDataset or
        # included in the condition tuple passed to the generator.
        "target": torch.tensor([row["target"] for row in rows], dtype=torch.float32),
    }


class VariantDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.variants = data["actions"].shape[1]

    def __len__(self):
        return len(self.data["actions"]) * self.variants

    def __getitem__(self, index):
        episode, variant = divmod(index, self.variants)
        return {
            "belief": self.data["belief"][episode],
            "relation": self.data["relation"][episode],
            "cue": self.data["cue"][episode],
            "centers": self.data["centers"][episode],
            "valid": self.data["valid"][episode],
            "pose_budget": self.data["pose_budget"][episode],
            "actions": self.data["actions"][episode, variant],
            "length": self.data["length"][episode, variant],
            "trajectory_slot": torch.tensor(variant, dtype=torch.long),
        }


def condition_from(batch, device):
    return tuple(batch[key].to(device, non_blocking=True) for key in (
        "belief", "relation", "cue", "centers", "valid", "pose_budget"))


def train(model, data, args, device):
    loader = DataLoader(
        VariantDataset(data), batch_size=args.batch_size, shuffle=True,
        pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    scheduler = make_ddpm_scheduler(args.diffusion_steps)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); epoch_loss, noise_loss, length_loss = [], [], []
        for batch in loader:
            clean = batch["actions"].to(device, non_blocking=True).float()
            timestep = torch.randint(
                args.diffusion_steps, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            noisy = scheduler.add_noise(clean, noise, timestep)
            output = model(
                noisy, timestep, *condition_from(batch, device),
                trajectory_slot=batch["trajectory_slot"].to(device))
            denoise = F.mse_loss(output["noise"], noise)
            length = F.cross_entropy(
                output["length_logits"], batch["length"].to(device))
            loss = denoise + 0.05 * length
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
            noise_loss.append(float(denoise.detach()))
            length_loss.append(float(length.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(epoch_loss)),
                  "noise_loss": float(np.mean(noise_loss)),
                  "length_loss": float(np.mean(length_loss))}
        history.append(record)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(record), flush=True)
    return history


def integrate_actions(actions, lengths, starts, protocol):
    """Turn generated normalized deltas into bounded map trajectories."""
    scale = protocol.maximum_translation_per_action_m / protocol.map_size_m
    batch, samples, horizon = actions.shape[:3]
    positions = torch.zeros(batch, samples, horizon + 1, 2)
    positions[:, :, 0] = starts[:, None]
    current = starts[:, None].expand(-1, samples, -1).clone()
    for step in range(horizon):
        delta = actions[:, :, step, :2].cpu()
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1.0)
        delta = delta / norm * scale
        active = step < lengths.cpu()
        current = torch.where(active[..., None], (current + delta).clamp(0, 1), current)
        positions[:, :, step + 1] = current
    return positions


@torch.inference_mode()
def evaluate(model, data, args, device):
    model.eval(); condition = tuple(data[key] for key in (
        "belief", "relation", "cue", "centers", "valid", "pose_budget"))
    sampled, length = ddim_sample(
        model, condition, samples=args.samples,
        diffusion_steps=args.diffusion_steps,
        inference_steps=args.inference_steps)
    raw_positions = integrate_actions(
        sampled, length, data["start"], OfficialProtocol())
    protocol = OfficialProtocol()
    rule_endpoint = data["rule_positions"][:, :args.samples, -1]
    positions, effective_length = bounded_endpoint_projection(
        sampled, length, data["start"], rule_endpoint,
        protocol.maximum_translation_per_action_m / protocol.map_size_m)
    generated_endpoint = positions[:, :, -1]
    raw_endpoint = raw_positions[:, :, -1]
    endpoint_distance = torch.linalg.vector_norm(
        generated_endpoint[:, :, None] - rule_endpoint[:, None], dim=-1) * 410.0
    assigned = endpoint_distance.argmin(-1)
    close = endpoint_distance.amin(-1).le(25.0)
    raw_rule_distance = torch.linalg.vector_norm(
        raw_endpoint[:, :, None] - data["rule_positions"][:, None, :, -1],
        dim=-1) * 410.0
    generated_target_distance = torch.linalg.vector_norm(
        generated_endpoint - data["target"][:, None], dim=-1) * 410.0
    rule_target_distance = torch.linalg.vector_norm(
        rule_endpoint - data["target"][:, None], dim=-1) * 410.0
    unique_modes = []
    for episode in range(len(assigned)):
        unique_modes.append(len(torch.unique(assigned[episode][close[episode]])))
    pairwise = torch.linalg.vector_norm(
        positions[:, :, None] - positions[:, None], dim=-1).mean(-1) * 410.0
    triangle = torch.triu(torch.ones(
        args.samples, args.samples, dtype=torch.bool), diagonal=1)
    return {
        "episodes": len(sampled),
        "samples_per_episode": args.samples,
        "mean_generated_length": float(effective_length.float().mean()),
        "minimum_generated_length": int(effective_length.min()),
        "maximum_generated_length": int(effective_length.max()),
        "paths_above_20_steps": int(effective_length.gt(20).sum()),
        "mean_pairwise_path_distance_m": float(pairwise[:, triangle].mean()),
        "mean_unique_rule_modes_within_25m": float(np.mean(unique_modes)),
        "episodes_covering_two_or_more_modes_percent": float(
            np.mean(np.asarray(unique_modes) >= 2) * 100),
        "generated_endpoints_within_25m_of_any_rule_percent": float(
            close.float().mean() * 100),
        "raw_unconstrained_endpoints_within_25m_of_any_rule_percent": float(
            raw_rule_distance.amin(-1).le(25).float().mean() * 100),
        "target_metric_only": {
            "generated_best_of_8_hit20": float(
                generated_target_distance.amin(-1).le(20).float().mean() * 100),
            "rule_best_of_8_hit20": float(
                rule_target_distance.amin(-1).le(20).float().mean() * 100),
            "generated_top1_unscored_hit20": float(
                generated_target_distance[:, 0].le(20).float().mean() * 100),
        },
        "anti_leakage": {
            "target_coordinate_in_model_inputs": False,
            "target_coordinate_use": "final metrics only",
            "supervision_source": "frozen belief modes plus current pose",
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
        },
    }


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_data(args, device, count=args.max_train)
    holdout = (build_data(args, device, start=args.max_train, count=args.holdout)
               if args.holdout else None)
    model = ConditionalUnet1D(base=args.base_channels).to(device)
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resumed["model"])
    parameter_count = sum(value.numel() for value in model.parameters())
    history = train(model, data, args, device)
    train_result = evaluate(model, data, args, device)
    holdout_result = evaluate(model, holdout, args, device) if holdout else None
    result = {
        "experiment": "target-free multi-trajectory diffusion generation gate",
        "train_episodes": args.max_train,
        "training_variants": int(data["actions"].shape[0] * data["actions"].shape[1]),
        "parameter_count": parameter_count,
        "first_epoch_loss": history[0]["loss"],
        "final_epoch_loss": history[-1]["loss"],
        "loss_reduction_percent": 100 * (1 - history[-1]["loss"] / history[0]["loss"]),
        "train_generation": train_result,
        "holdout_generation": holdout_result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args),
                "history": history}, args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
