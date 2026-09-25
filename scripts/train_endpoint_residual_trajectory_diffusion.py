#!/usr/bin/env python3
"""Train endpoint-conditioned lateral-residual trajectory diffusion V2."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import OfficialProtocol
from independent_nav.trajectory_diffusion import (
    EndpointResidualUnet1D, ddim_sample_endpoint_residual,
    endpoint_condition_features, lateral_residual_targets,
    make_ddpm_scheduler,
)
from scripts.train_trajectory_diffusion import build_data
from scripts.evaluate_official_static_belief import DEFAULT_SOURCE


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
        "runs/trajectory_diffusion_endpoint_residual_train1024_holdout256"))
    parser.add_argument("--max-train", type=int, default=1024)
    parser.add_argument("--holdout", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def add_residual_targets(data, protocol):
    endpoints = data["rule_positions"][:, :8, -1]
    residual, minimum, extra, effective = lateral_residual_targets(
        data["rule_positions"][:, :8], data["length"][:, :8],
        data["start"], endpoints,
        protocol.maximum_translation_per_action_m / protocol.map_size_m)
    data = dict(data)
    data.update({
        "endpoint": endpoints,
        "residual": residual,
        "minimum_steps": minimum,
        "extra_steps": extra,
        "effective_steps": effective,
        "endpoint_features": endpoint_condition_features(
            data["start"], endpoints, minimum,
            torch.full((len(endpoints),), protocol.max_action_len)),
    })
    return data


class ResidualDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.variants = data["endpoint"].shape[1]

    def __len__(self):
        return len(self.data["endpoint"]) * self.variants

    def __getitem__(self, index):
        episode, slot = divmod(index, self.variants)
        return {
            "belief": self.data["belief"][episode],
            "relation": self.data["relation"][episode],
            "cue": self.data["cue"][episode],
            "centers": self.data["centers"][episode],
            "valid": self.data["valid"][episode],
            "pose_budget": self.data["pose_budget"][episode],
            "endpoint_features": self.data["endpoint_features"][episode, slot],
            "residual": self.data["residual"][episode, slot],
            "extra_steps": self.data["extra_steps"][episode, slot],
            "trajectory_slot": torch.tensor(slot, dtype=torch.long),
        }


def scene_condition(batch, device):
    return tuple(batch[key].to(device, non_blocking=True) for key in (
        "belief", "relation", "cue", "centers", "valid", "pose_budget"))


def train(model, data, args, device):
    loader = DataLoader(
        ResidualDataset(data), batch_size=args.batch_size, shuffle=True,
        pin_memory=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = make_ddpm_scheduler(args.diffusion_steps)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses, noises, extras = [], [], []
        for batch in loader:
            clean = batch["residual"].to(device, non_blocking=True).float()
            timestep = torch.randint(
                args.diffusion_steps, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            noisy = scheduler.add_noise(clean, noise, timestep)
            output = model(
                noisy, timestep, *scene_condition(batch, device),
                batch["endpoint_features"].to(device, non_blocking=True),
                trajectory_slot=batch["trajectory_slot"].to(device))
            noise_loss = F.mse_loss(output["noise"], noise)
            extra_loss = F.cross_entropy(
                output["extra_step_logits"],
                batch["extra_steps"].to(device))
            loss = noise_loss + 0.05 * extra_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            noises.append(float(noise_loss.detach()))
            extras.append(float(extra_loss.detach()))
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "noise_loss": float(np.mean(noises)),
            "extra_step_loss": float(np.mean(extras)),
        }
        history.append(record)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(record), flush=True)
    return history


@torch.inference_mode()
def evaluate(model, data, args, device, protocol):
    model.eval()
    scene = tuple(data[key] for key in (
        "belief", "relation", "cue", "centers", "valid", "pose_budget"))
    positions, effective, extra = ddim_sample_endpoint_residual(
        model, scene, data["endpoint"], data["start"],
        data["minimum_steps"],
        torch.full((len(data["start"]),), protocol.max_action_len),
        samples=8, diffusion_steps=args.diffusion_steps,
        inference_steps=args.inference_steps,
        step_scale=(protocol.maximum_translation_per_action_m /
                    protocol.map_size_m))
    endpoint_error = torch.linalg.vector_norm(
        positions[:, :, -1] - data["endpoint"], dim=-1) * protocol.map_size_m
    segment = torch.linalg.vector_norm(
        positions[:, :, 1:] - positions[:, :, :-1], dim=-1)
    valid = torch.arange(model.horizon)[None, None] < effective[..., None]
    arc = (segment * valid).sum(-1) * protocol.map_size_m
    direct = torch.linalg.vector_norm(
        data["endpoint"] - data["start"][:, None], dim=-1) * protocol.map_size_m
    ratio = arc / direct.clamp_min(1.0)
    distance = torch.linalg.vector_norm(
        positions - data["endpoint"][:, :, None], dim=-1)
    reverse = ((distance[:, :, 1:] - distance[:, :, :-1]) > 1e-5) & valid
    target_distance = torch.linalg.vector_norm(
        data["endpoint"] - data["target"][:, None], dim=-1) * protocol.map_size_m
    return {
        "episodes": len(positions),
        "endpoint_error_mean_m": float(endpoint_error.mean()),
        "endpoint_error_max_m": float(endpoint_error.max()),
        "path_efficiency_ratio_mean": float(ratio.mean()),
        "path_efficiency_ratio_median": float(ratio.median()),
        "paths_over_110_percent": float(ratio.gt(1.10).float().mean() * 100),
        "reverse_progress_steps_percent": float(
            reverse.float().sum() / valid.float().sum().clamp_min(1) * 100),
        "mean_extra_steps": float(extra.float().mean()),
        "paths_above_20_steps": int(effective.gt(20).sum()),
        "target_metric_only_best_of_8_hit20": float(
            target_distance.amin(-1).le(20).float().mean() * 100),
        "anti_leakage": {
            "target_in_model_input": False,
            "endpoint_source": "frozen belief hypotheses",
        },
    }


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = OfficialProtocol()
    train_data = add_residual_targets(
        build_data(args, device, count=args.max_train), protocol)
    holdout_data = add_residual_targets(
        build_data(args, device, start=args.max_train, count=args.holdout),
        protocol) if args.holdout else None
    model = EndpointResidualUnet1D(base=args.base_channels).to(device)
    history = train(model, train_data, args, device)
    result = {
        "experiment": "endpoint-conditioned shortest-path residual diffusion",
        "architecture": "endpoint_residual_v2",
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "train_episodes": args.max_train,
        "holdout_episodes": args.holdout,
        "first_epoch_loss": history[0]["loss"],
        "final_epoch_loss": history[-1]["loss"],
        "train_generation": evaluate(model, train_data, args, device, protocol),
        "holdout_generation": (
            evaluate(model, holdout_data, args, device, protocol)
            if holdout_data else None),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "architecture": "endpoint_residual_v2",
        "config": vars(args),
        "history": history,
    }, args.output_dir / "model.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
