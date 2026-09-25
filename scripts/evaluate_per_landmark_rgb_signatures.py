#!/usr/bin/env python3
"""Evaluate instance-preserving RGB relation signatures for three candidates."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_nav.belief_transformer import BeliefTransformer, TemporalBeliefTransformer
from scripts.build_road_structure_layer import build_binary_layer
from scripts.evaluate_rgb_relation_signatures import (
    DEFAULT_SOURCE, candidate_masks, corridor_mask, expected_landmark_classes,
    load_rows, predicted_types, road_required, top_fraction, type_catalog,
)
from scripts.train_belief_transformer import BeliefDataset
from scripts.train_relation_geometry_ranker import (
    corrected_landmark_geometry, opposite_direction_cues,
)
from scripts.train_rgb_candidate_ranker import metrics
from scripts.train_temporal_belief_transformer import same_map_shuffle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, default=Path(
        "runs/temporal_belief_candidate_rgb_gate5_large/semantic_bev_support20_grid64.pt"))
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_SOURCE /
                        "candidate_rgb_data/geometry_top5_oof_grouped_enriched")
    parser.add_argument("--relation-cache", type=Path, default=DEFAULT_SOURCE /
                        "relation_geometry_ranker_contrastive/relation_features.pt")
    parser.add_argument("--type-checkpoint", type=Path, default=Path(
        "runs/temporal_belief_transformer_gate5_large/model.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/per_landmark_rgb_signature_gate8/results.json"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def extract_instance_signatures(rows, base, cache, predicted_type, objects, catalog):
    count, candidates, landmarks = len(rows), 3, base.valid.shape[1]
    candidate_feature = torch.zeros(count, candidates, 4)
    visual = torch.zeros(count, candidates, landmarks, 4)
    distances = torch.zeros(count, candidates)
    road_layers = {}
    for index, row in enumerate(rows):
        semantic = cache["semantic"][index].float()
        visible = cache["visibility"][index]
        grid_size = semantic.shape[-1]
        masks = candidate_masks(row["candidates"], grid_size=grid_size)
        expected = expected_landmark_classes(row, catalog)
        name = row["map_name"]
        if name not in road_layers:
            value, _ = build_binary_layer(
                objects, name, resolution=grid_size,
                object_types=("TrafficRoad", "Footpath"))
            road_layers[name] = torch.from_numpy(value.astype(bool))
        required = road_required(row["description"])
        target_class = int(predicted_type[index]) + 1
        for candidate_index in range(candidates):
            view = min(candidate_index, semantic.shape[0] - 1)
            local = masks[candidate_index] & visible[view]
            target = semantic[view, target_class]
            target_mean = float(target[local].mean()) if local.any() else 0.0
            target_top = top_fraction(target, local)
            candidate_feature[index, candidate_index] = torch.tensor([
                -candidate_index / 2.0, target_mean, target_top,
                float(local.float().mean()),
            ])
            for landmark_index in range(landmarks):
                if not base.valid[index, landmark_index]:
                    continue
                mask = base.instance_masks[index, landmark_index]
                landmark_visible = mask & visible[view]
                coverage = float(
                    landmark_visible.sum().float() / mask.sum().clamp_min(1))
                semantic_class = (expected[landmark_index]
                                  if landmark_index < len(expected) else 1)
                agreement = (float(semantic[view, semantic_class][landmark_visible].mean())
                             if landmark_visible.any() else 0.0)
                confirmation = coverage * agreement
                corridor = corridor_mask(
                    row["candidates"][candidate_index],
                    base.centers[index, landmark_index:landmark_index + 1],
                    base.valid[index, landmark_index:landmark_index + 1],
                    grid_size=grid_size)
                expected_road = corridor & visible[view] & road_layers[name]
                road_confirmation = (
                    float(semantic[view, 3][expected_road].mean())
                    if required and expected_road.any() else (1.0 if not required else 0.0))
                visual[index, candidate_index, landmark_index] = torch.tensor([
                    coverage, confirmation, target_top * confirmation,
                    road_confirmation,
                ])
        target_xy = torch.tensor(row["target"])
        candidate_xy = torch.tensor(row["candidates"][:candidates])
        distances[index] = torch.linalg.vector_norm(
            candidate_xy - target_xy[None], dim=-1) * 410.0
    geometry, valid = corrected_landmark_geometry(rows, max_landmarks=landmarks)
    return {
        "candidate": candidate_feature,
        "visual": visual,
        "geometry": geometry[:, :candidates],
        "relation": base.relation,
        "cue": base.cues,
        "opposite_cue": opposite_direction_cues(base.cues),
        "valid": valid,
        "distance": distances,
    }


class InstanceSignatureRanker(nn.Module):
    def __init__(self, visual_dim, hidden=64):
        super().__init__()
        self.visual_dim = visual_dim
        self.candidate = nn.Sequential(
            nn.Linear(4, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.relation = nn.Sequential(
            nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.cue = nn.Sequential(
            nn.Linear(12, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.geometry = nn.Sequential(
            nn.Linear(10, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.visual = (nn.Sequential(
            nn.Linear(visual_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
            if visual_dim else None)
        layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 2, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, 2, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, candidate, relation, cue, geometry, visual, valid):
        batch, candidates, landmarks = geometry.shape[:3]
        candidate_token = self.candidate(candidate).reshape(batch * candidates, 1, -1)
        token = (self.relation(relation.float())[:, None]
                 + self.cue(cue.float())[:, None]
                 + self.geometry(geometry.float()))
        if self.visual is not None:
            token = token + self.visual(visual[..., :self.visual_dim].float())
        token = token.reshape(batch * candidates, landmarks, -1)
        sequence = torch.cat((candidate_token, token), dim=1)
        padding = torch.cat((
            torch.zeros(batch, 1, dtype=torch.bool, device=valid.device), ~valid), -1)
        padding = padding[:, None].expand(batch, candidates, landmarks + 1).reshape(
            batch * candidates, landmarks + 1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding)
        return self.score(encoded[:, 0]).reshape(batch, candidates)


def loss_function(score, distance):
    target = F.softmax(-distance / 20.0, dim=-1)
    return -(target * F.log_softmax(score, dim=-1)).sum(-1).mean()


def batches(data, batch_size, shuffle):
    return DataLoader(TensorDataset(
        data["candidate"], data["relation"], data["cue"], data["geometry"],
        data["visual"], data["valid"], data["distance"]),
        batch_size=batch_size, shuffle=shuffle)


@torch.inference_mode()
def scores(model, data, device, mode="normal"):
    output = []
    cue = data["opposite_cue"] if mode == "opposite_relation" else data["cue"]
    visual = data["visual"]
    if mode == "wrong_landmark_visual":
        # Preserve candidate RGB/type while binding landmark evidence from a
        # different same-map episode, supplied by the caller.
        visual = data["wrong_visual"]
    dataset = TensorDataset(
        data["candidate"], data["relation"], cue, data["geometry"],
        visual, data["valid"])
    for values in DataLoader(dataset, batch_size=256):
        output.append(model(*(value.to(device) for value in values)).cpu())
    return torch.cat(output)


def train_model(train, val, visual_dim, epochs, seed, device):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = InstanceSignatureRanker(visual_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    best, state, best_epoch = None, None, None
    for epoch in range(1, epochs + 1):
        model.train()
        for candidate, relation, cue, geometry, visual, valid, distance in batches(
                train, 128, True):
            score = model(candidate.to(device), relation.to(device), cue.to(device),
                          geometry.to(device), visual.to(device), valid.to(device))
            loss = loss_function(score, distance.to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        prediction = scores(model, val, device)
        value = metrics(val["distance"], prediction.argmax(-1))
        key = value["hit20"], -value["mean_ne"]
        if best is None or key > best:
            best, best_epoch = key, epoch
            state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model.eval(), best_epoch


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relation = torch.load(args.relation_cache, map_location="cpu", weights_only=True)
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    objects = json.loads((args.data_root / "cityrefer" / "objects.json").read_text())
    catalog = type_catalog(objects)
    grid_size = int(cache["metadata"]["grid_size"])
    rows, base = {}, {}
    for split in ("train_seen", "val_seen"):
        all_rows = load_rows(args.candidate_dir / f"{split}.jsonl")
        source = cache["splits"][split]["source_rows"].tolist()
        rows[split] = [all_rows[index] for index in source]
        base[split] = BeliefDataset(rows[split], relation[split][source], grid_size, 9)

    static = BeliefTransformer(grid_size=20, hidden=96, layers=2)
    type_model = TemporalBeliefTransformer(static, hidden=64).to(device)
    type_model.load_state_dict(torch.load(
        args.type_checkpoint, map_location="cpu", weights_only=True)["model"])
    type_model.eval()
    predicted = {split: predicted_types(type_model, base[split], device)
                 for split in rows}
    data = {split: extract_instance_signatures(
        rows[split], base[split], cache["splits"][split], predicted[split],
        objects, catalog) for split in rows}
    for split in rows:
        switched = same_map_shuffle(cache["splits"][split]["map_names"])
        data[split]["wrong_visual"] = data[split]["visual"][switched]

    variants = {
        "relation_geometry_no_rgb_landmarks": 0,
        "plus_landmark_visibility_confirmation": 3,
        "plus_per_landmark_road_confirmation": 4,
    }
    result = {
        "counts": {split: len(rows[split]) for split in rows},
        "candidate_prior": metrics(
            data["val_seen"]["distance"],
            torch.zeros(len(rows["val_seen"]), dtype=torch.long)),
        "top3_oracle": metrics(
            data["val_seen"]["distance"], data["val_seen"]["distance"].argmin(-1)),
        "variants": {},
        "anti_leakage": {
            "training": "train_seen only", "selection": "val_seen only",
            "val_unseen_loaded": False, "test_unseen_loaded": False,
            "target_coordinates": "listwise labels and final metrics only",
            "landmark_representation": "up to nine independent tokens; no averaging",
        },
    }
    for name, dimension in variants.items():
        model, epoch = train_model(
            data["train_seen"], data["val_seen"], dimension,
            args.epochs, args.seed, device)
        normal_score = scores(model, data["val_seen"], device)
        entry = {"selected_epoch": epoch,
                 "normal": metrics(data["val_seen"]["distance"], normal_score.argmax(-1))}
        if dimension:
            wrong = scores(model, data["val_seen"], device, "wrong_landmark_visual")
            opposite = scores(model, data["val_seen"], device, "opposite_relation")
            entry["wrong_landmark_visual"] = metrics(
                data["val_seen"]["distance"], wrong.argmax(-1))
            entry["opposite_relation"] = metrics(
                data["val_seen"]["distance"], opposite.argmax(-1))
        result["variants"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
