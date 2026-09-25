#!/usr/bin/env python3
"""Train an interpretable cue-by-cue candidate/landmark relation ranker."""

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, build_split, evaluate, listwise_loss_per_row, load_rows,
    ranking_loss, scores,
)
from scripts.train_rgb_candidate_ranker import metrics


ROOT = Path("runs")


class ExplicitRelationRanker(nn.Module):
    """Score named relations explicitly, then aggregate landmark constraints."""

    def __init__(self, cue_dim=len(CUE_NAMES), hidden=64):
        super().__init__()
        self.cue_dim = cue_dim
        # Compatibility marker used by the shared controlled-ablation helper.
        self.cue = nn.Identity()
        self.geometry_to_cues = nn.Sequential(
            nn.Linear(10, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, cue_dim))
        # BERT is used only as a fallback when the explicit parser found no cue.
        self.fallback_language = nn.Sequential(
            nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.fallback_geometry = nn.Sequential(
            nn.Linear(10, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.fallback_score = nn.Linear(hidden, 1)
        self.aggregate = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, relation, geometry, valid, cue=None):
        if cue is None:
            raise ValueError("explicit cue tensor is required")
        cue = cue.float()
        geometry = geometry.float()
        satisfaction = self.geometry_to_cues(geometry)
        cue_count = cue.sum(-1).clamp_min(1.0)
        explicit = (satisfaction * cue[:, None]).sum(-1) / cue_count[:, None]
        fallback_mask = cue.sum(-1).eq(0)
        language = self.fallback_language(relation.float())[:, None]
        spatial = self.fallback_geometry(geometry)
        fallback = self.fallback_score(language * spatial).squeeze(-1)
        pair = torch.where(fallback_mask[:, None], fallback, explicit)
        pair = pair.masked_fill(~valid[:, None], 0.0)
        count = valid.sum(-1).clamp_min(1).float()[:, None]
        mean = pair.sum(-1) / count
        minimum = pair.masked_fill(~valid[:, None], float("inf")).amin(-1)
        maximum = pair.masked_fill(~valid[:, None], -float("inf")).amax(-1)
        no_landmark = valid.sum(-1).eq(0)[:, None]
        minimum = torch.where(no_landmark, torch.zeros_like(minimum), minimum)
        maximum = torch.where(no_landmark, torch.zeros_like(maximum), maximum)
        return self.aggregate(torch.stack((mean, minimum, maximum), -1)).squeeze(-1)


def train_one(train, validation, seed, device, epochs=15):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ExplicitRelationRanker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    dataset = TensorDataset(
        train["relation"], train["geometry"], train["valid"], train["cue"],
        train["distance"], train["switched_relation"], train["switched_cue"],
        train["switched_geometry"], train["switched_valid"], train["switch_changed"])
    best, best_key, best_epoch = None, None, 0
    for epoch in range(epochs):
        model.train()
        for (relation, geometry, valid, cue, distance, switched_relation,
             switched_cue, switched_geometry, switched_valid, switch_changed) in DataLoader(
                dataset, batch_size=256, shuffle=True, pin_memory=True):
            relation = relation.to(device); geometry = geometry.to(device)
            valid = valid.to(device); cue = cue.to(device); distance = distance.to(device)
            score = model(relation, geometry, valid, cue)
            loss = ranking_loss(score, distance, 0.2)[0]
            if switch_changed.any():
                selected = switch_changed.to(device)
                wrong_relation = model(
                    switched_relation.to(device), geometry, valid,
                    switched_cue.to(device))
                wrong_landmark = model(
                    relation, switched_geometry.to(device), switched_valid.to(device), cue)
                correct_ce = listwise_loss_per_row(score, distance)
                relation_ce = listwise_loss_per_row(wrong_relation, distance)
                landmark_ce = listwise_loss_per_row(wrong_landmark, distance)
                mismatch = 0.5 * (
                    F.softplus(0.2 + correct_ce[selected] - relation_ce[selected])
                    + F.softplus(0.2 + correct_ce[selected] - landmark_ce[selected]))
                loss = loss + 0.2 * mismatch.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = scores(model, validation, device)
            value = metrics(validation["distance"], prediction.argmax(-1))
        key = (value["hit20"], -value["mean_ne"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch + 1
            best = {name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()}
        print(f"explicit seed={seed} epoch={epoch + 1}/{epochs} "
              f"val_hit20={value['hit20']:.2f}", flush=True)
    model.load_state_dict(best)
    return model.eval(), best_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    relation_dir = ROOT / "relation_geometry_ranker_contrastive"
    output_dir = ROOT / "explicit_relation_ranker"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {split: load_rows(candidate_dir / f"{split}.jsonl", 5)
            for split in ("train_seen", "val_seen", "val_unseen")}
    rows["test_unseen"] = load_rows(ROOT / "test_unseen_fusion/test_unseen.jsonl", 5)
    relation_cache = torch.load(
        relation_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    relation = {split: relation_cache[split]
                for split in ("train_seen", "val_seen", "val_unseen")}
    relation["test_unseen"] = torch.load(
        ROOT / "test_unseen_fusion/relation_features.pt",
        map_location="cpu", weights_only=True)
    data = {split: build_split(rows[split], relation[split], 9) for split in rows}
    runs = []
    for seed in (0, 1, 2):
        model, epoch = train_one(data["train_seen"], data["val_seen"], seed, device)
        run = {"seed": seed, "best_epoch": epoch}
        for split in ("val_unseen", "test_unseen"):
            run[split] = evaluate(model, data[split], device, prior_weight=0.0)
        runs.append(run)
        torch.save(model.state_dict(), output_dir / f"model_seed{seed}.pt")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    keys = ("learned_relation", "wrong_relation_same_map", "wrong_landmark_same_map",
            "no_relation", "no_structured_cue", "opposite_direction",
            "candidate_prior_top1", "oracle_top5")
    mean = {}
    for split in ("val_unseen", "test_unseen"):
        mean[split] = {
            key: {metric: sum(run[split][key][metric] for run in runs) / len(runs)
                  for metric in ("mean_ne", "median_ne", "hit20", "hit50")}
            for key in keys}
    result = {
        "architecture": "per-cue geometry satisfaction plus mean/min/max landmark aggregation",
        "cue_names": CUE_NAMES,
        "runs": runs,
        "mean": mean,
        "anti_leakage": {
            "training": "train_seen only",
            "checkpoint_selection": "val_seen only",
            "test_used_for_training_or_selection": False,
            "target_coordinates": "ranking labels and final metrics only",
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
