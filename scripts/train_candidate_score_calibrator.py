#!/usr/bin/env python3
"""Train a small leakage-controlled calibrator over frozen candidate scores."""

import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_dualscale_appearance_fusion import direct_similarity
from scripts.fuse_relation_type_scores import normalized, type_compatibility
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES, RelationGeometryRanker, build_split, load_rows,
    scores as relation_scores,
)
from scripts.train_rgb_candidate_ranker import metrics


ROOT = Path("runs")


class CandidateCalibrator(nn.Module):
    def __init__(self, component_count, context_dim):
        super().__init__()
        input_dim = component_count + component_count * 3 + context_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, component, context):
        count = component.shape[1]
        summary = torch.cat((component.mean(1), component.amax(1),
                             component.std(1)), dim=-1)
        row = torch.cat((summary, context), dim=-1)[:, None].expand(-1, count, -1)
        return self.network(torch.cat((component, row), dim=-1)).squeeze(-1)


def split_maps(rows):
    maps = sorted({row["map_name"] for row in rows})
    bucket = {name: int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 8
              for name in maps}
    test_maps = {name for name in maps if bucket[name] in (0, 4)}
    validation_maps = {name for name in maps if bucket[name] in (1, 5)}
    test = torch.tensor([row["map_name"] in test_maps for row in rows])
    validation = torch.tensor([row["map_name"] in validation_maps for row in rows])
    train = ~(test | validation)
    return train, validation, test, sorted(validation_maps), sorted(test_maps)


def assemble_frozen_features(device):
    candidate_dir = ROOT / "candidate_rgb_data/geometry_top5_oof_grouped_enriched"
    relation_dir = ROOT / "relation_geometry_ranker_contrastive"
    test_dir = ROOT / "test_unseen_fusion"
    rows = {
        "val_seen": load_rows(candidate_dir / "val_seen.jsonl", 5),
        "test_unseen": load_rows(test_dir / "test_unseen.jsonl", 5),
    }
    relation_feature = torch.load(
        relation_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    test_relation = torch.load(
        test_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    data = {
        "val_seen": build_split(rows["val_seen"], relation_feature["val_seen"], 9),
        "test_unseen": build_split(rows["test_unseen"], test_relation, 9),
    }
    type_feature = torch.load(
        ROOT / "target_type_alignment/features.pt", map_location="cpu", weights_only=True)
    scale40 = torch.load(
        ROOT / "target_type_alignment/multiscale/scale_40.pt",
        map_location="cpu", weights_only=True)
    candidate40 = {
        "val_seen": torch.load(
            ROOT / "relation_type_fusion/scale40_candidate_features.pt",
            map_location="cpu", weights_only=True)["val_seen"],
        "test_unseen": torch.load(
            test_dir / "siglip40_features.pt", map_location="cpu", weights_only=True)["candidate"],
    }
    test40 = torch.load(
        test_dir / "siglip40_features.pt", map_location="cpu", weights_only=True)
    text = {
        "val_seen": type_feature["text"]["val_seen"]["features"],
        "test_unseen": test40["text"],
    }
    candidate20 = torch.load(
        test_dir / "siglip20_candidate_features.pt", map_location="cpu", weights_only=True)
    collected = {split: {"relation": [], "type": [], "text_probability": []}
                 for split in rows}
    for seed in (0, 1, 2):
        text_model = train_probe(type_feature["text"]["train_seen"],
                                 type_feature["text"]["val_seen"], seed, 40, device)
        image_model = train_probe(scale40["target"]["train_seen"],
                                  scale40["target"]["val_seen"], seed, 40, device)
        relation_model = RelationGeometryRanker(cue_dim=len(CUE_NAMES)).to(device)
        relation_model.load_state_dict(torch.load(
            relation_dir / f"model_seed{seed}.pt", map_location="cpu", weights_only=True))
        relation_model.eval()
        for split in rows:
            collected[split]["relation"].append(normalized(
                relation_scores(relation_model, data[split], device)))
            type_score, text_logits, _ = type_compatibility(
                text_model, image_model, text[split], candidate40[split], device)
            collected[split]["type"].append(normalized(type_score))
            collected[split]["text_probability"].append(
                F.softmax(text_logits, dim=-1))
        del text_model, image_model, relation_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {}
    for split in rows:
        prior = normalized(-torch.arange(5, dtype=torch.float32)[None].expand(
            len(rows[split]), 5))
        components = torch.stack((
            torch.stack(collected[split]["relation"]).mean(0),
            torch.stack(collected[split]["type"]).mean(0),
            direct_similarity(text[split], candidate40[split]),
            direct_similarity(text[split], candidate20[split]),
            prior,
        ), dim=-1)
        cue = data[split]["cue"].amax(1)
        landmark = torch.stack((data[split]["landmark_count"].eq(1),
                                data[split]["landmark_count"].eq(2),
                                data[split]["landmark_count"].ge(3)), dim=-1).float()
        text_probability = torch.stack(collected[split]["text_probability"]).mean(0)
        context = torch.cat((cue, landmark, text_probability), dim=-1)
        result[split] = {"component": components, "context": context,
                         "distance": data[split]["distance"]}
    return rows, result


def train_once(data, train_mask, validation_mask, seed, epochs=160):
    random.seed(seed)
    torch.manual_seed(seed)
    model = CandidateCalibrator(data["component"].shape[-1],
                                data["context"].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-2)
    indices = torch.where(train_mask)[0]
    loader = DataLoader(TensorDataset(indices), batch_size=256, shuffle=True)
    best, best_key, best_epoch = None, None, None
    for epoch in range(epochs):
        model.train()
        for (batch,) in loader:
            score = model(data["component"][batch], data["context"][batch])
            distance = data["distance"][batch]
            soft_target = F.softmax(-distance / 20.0, dim=-1)
            loss = -(soft_target * F.log_softmax(score, dim=-1)).sum(-1).mean()
            hit = distance.le(20.0).float()
            loss = loss + 0.15 * F.binary_cross_entropy_with_logits(score, hit)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            score = model(data["component"], data["context"])
            value = metrics(data["distance"][validation_mask],
                            score[validation_mask].argmax(-1))
        key = (value["hit20"], -value["mean_ne"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch + 1
            best = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best)
    return model.eval(), best_epoch


def train_fixed_epochs(data, seed, epochs):
    # Test labels are never involved. The epoch count comes from map validation,
    # then all val_seen rows are used for the final exploratory fit.
    all_rows = torch.ones(len(data["distance"]), dtype=torch.bool)
    random.seed(seed)
    torch.manual_seed(seed)
    model = CandidateCalibrator(data["component"].shape[-1],
                                data["context"].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-2)
    loader = DataLoader(TensorDataset(torch.where(all_rows)[0]), batch_size=256, shuffle=True)
    for _ in range(epochs):
        model.train()
        for (batch,) in loader:
            score = model(data["component"][batch], data["context"][batch])
            distance = data["distance"][batch]
            target = F.softmax(-distance / 20.0, dim=-1)
            loss = -(target * F.log_softmax(score, dim=-1)).sum(-1).mean()
            loss += 0.15 * F.binary_cross_entropy_with_logits(
                score, distance.le(20.0).float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, data = assemble_frozen_features(device)
    train_mask, validation_mask, test_mask, validation_maps, test_maps = split_maps(
        rows["val_seen"])
    holdout_runs, final_runs = [], []
    for seed in (0, 1, 2):
        model, epoch = train_once(data["val_seen"], train_mask, validation_mask, seed)
        with torch.inference_mode():
            rank = model(data["val_seen"]["component"],
                         data["val_seen"]["context"]).argmax(-1)
        holdout_runs.append({"seed": seed, "selected_epoch": epoch,
                             "metrics": metrics(data["val_seen"]["distance"][test_mask],
                                                rank[test_mask])})
        final = train_fixed_epochs(data["val_seen"], seed, epoch)
        with torch.inference_mode():
            rank = final(data["test_unseen"]["component"],
                         data["test_unseen"]["context"]).argmax(-1)
        final_runs.append({"seed": seed, "selected_epoch": epoch,
                           "metrics": metrics(data["test_unseen"]["distance"], rank)})

    def mean(runs):
        return {key: sum(run["metrics"][key] for run in runs) / len(runs)
                for key in ("mean_ne", "median_ne", "hit20", "hit50")}

    result = {
        "architecture": "32-hidden-unit candidate score calibrator",
        "inputs": ["relation", "type", "appearance40", "appearance20", "prior",
                   "score summaries", "relation cues", "landmark count", "text type probabilities"],
        "map_partition": {
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "heldout_rows": int(test_mask.sum()),
            "validation_maps": validation_maps,
            "heldout_maps": test_maps,
        },
        "map_holdout_runs": holdout_runs,
        "map_holdout_mean": mean(holdout_runs),
        "test_unseen_exploratory_runs": final_runs,
        "test_unseen_exploratory_mean": mean(final_runs),
        "anti_leakage": {
            "test_unseen_targets_used_for_training_selection_or_early_stopping": False,
            "test_unseen_targets_used_for": "final exploratory metric only",
        },
    }
    path = ROOT / "test_unseen_fusion/candidate_calibrator_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
