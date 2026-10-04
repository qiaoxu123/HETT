"""Diagnose landmark alignment, target understanding, and scene-switch tracking.

This script intentionally does NOT run HETT Stage-1/Stage-2 navigation. It tests
whether the landmark-centric multimodal map itself has learned/retained the right
identity across language, top-view RGB, ego-view RGB, and abrupt scene switches.

Metrics:
  1) Text -> landmark target Recall@1/5 (lexical map retrieval)
  2) Landmark name -> top RGB Recall@1/5 (SigLIP)
  3) Ego RGB -> landmark name Recall@1/5 (SigLIP)
  4) Ego RGB -> top RGB landmark ID Recall@1/5 (SigLIP)
  5) Same-target tracking accuracy / consecutive consistency
  6) Scene-switch first-frame accuracy / stale-target rate
"""
import argparse
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.landmark_multimodal_map import LandmarkMultimodalMap


def _resolve(cache_path: Path, asset_path):
    if not asset_path:
        return None
    path = Path(asset_path)
    return path if path.is_absolute() else cache_path.parent / path


class SigLIPEncoder:
    def __init__(self, model_name, device):
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode_text(self, texts, batch_size=64):
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = self.processor(
                text=batch,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            features = self.model.get_text_features(**inputs)
            chunks.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def encode_images(self, image_paths, batch_size=32):
        chunks = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            chunks.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(chunks, dim=0)


def recall_at_k(similarity, query_ids, candidate_ids, k):
    k = min(int(k), similarity.shape[1])
    indices = similarity.topk(k, dim=1).indices.numpy()
    hits = 0
    for row, target_id in zip(indices, query_ids):
        if any(candidate_ids[index] == target_id for index in row):
            hits += 1
    return hits / max(len(query_ids), 1)


def evaluate_text_target(semantic_map, split, altitude):
    total = hit1 = hit5 = matched = 0
    for trajectory in load_mturk_trajectories(split, "all", altitude):
        if not trajectory.descriptions:
            continue
        retrieved = semantic_map.retrieve_text(
            trajectory.descriptions[0],
            trajectory.map_name,
            topk=5,
        )
        ids = [record.landmark_id for record, _ in retrieved]
        target = int(trajectory.object_id)
        total += 1
        matched += int(bool(ids))
        hit1 += int(bool(ids) and ids[0] == target)
        hit5 += int(target in ids)
    denom = max(total, 1)
    return {
        "samples": total,
        "matched_rate": matched / denom,
        "recall@1": hit1 / denom,
        "recall@5": hit5 / denom,
    }


def build_visual_index(semantic_map, cache_path):
    top = []
    ego = []
    for record in semantic_map.records:
        top_path = _resolve(cache_path, record.top_rgb_path)
        if top_path and top_path.is_file():
            top.append((record, top_path))
        for view_index, view in enumerate(record.ego_views):
            ego_path = _resolve(cache_path, view.get("rgb_path"))
            if ego_path and ego_path.is_file():
                ego.append((record, view_index, view, ego_path))
    return top, ego


def per_map_similarity(query_features, query_records, candidate_features, candidate_records):
    rows = []
    candidate_ids = []
    for query_index, query_record in enumerate(query_records):
        valid = [
            index for index, record in enumerate(candidate_records)
            if record.map_name == query_record.map_name
        ]
        if not valid:
            continue
        sim = query_features[query_index:query_index + 1] @ candidate_features[valid].T
        rows.append((query_record, sim.squeeze(0), valid))
        candidate_ids.append([candidate_records[index].landmark_id for index in valid])
    return rows, candidate_ids


def evaluate_name_to_top(encoder, top):
    if not top:
        return {}
    records = [item[0] for item in top]
    paths = [item[1] for item in top]
    text_features = encoder.encode_text([record.name for record in records])
    image_features = encoder.encode_images(paths)

    total = hit1 = hit5 = 0
    for i, record in enumerate(records):
        valid = [j for j, other in enumerate(records) if other.map_name == record.map_name]
        sim = text_features[i:i + 1] @ image_features[valid].T
        ranked = sim.squeeze(0).argsort(descending=True)
        ids = [records[valid[j]].landmark_id for j in ranked[:5].tolist()]
        total += 1
        hit1 += int(ids and ids[0] == record.landmark_id)
        hit5 += int(record.landmark_id in ids)
    denom = max(total, 1)
    return {"samples": total, "recall@1": hit1 / denom, "recall@5": hit5 / denom}


def evaluate_ego_to_name_and_top(encoder, top, ego):
    if not ego:
        return {}, {}, None, None, None

    name_records = [item[0] for item in top] if top else []
    top_paths = [item[1] for item in top]
    top_features = encoder.encode_images(top_paths) if top else None
    name_features = encoder.encode_text([record.name for record in name_records]) if top else None

    ego_records = [item[0] for item in ego]
    ego_paths = [item[3] for item in ego]
    ego_features = encoder.encode_images(ego_paths)

    def _eval(candidate_features):
        if candidate_features is None:
            return {}
        total = hit1 = hit5 = 0
        for i, record in enumerate(ego_records):
            valid = [
                j for j, other in enumerate(name_records)
                if other.map_name == record.map_name
            ]
            if not valid:
                continue
            sim = ego_features[i:i + 1] @ candidate_features[valid].T
            ranked = sim.squeeze(0).argsort(descending=True)
            ids = [name_records[valid[j]].landmark_id for j in ranked[:5].tolist()]
            total += 1
            hit1 += int(ids and ids[0] == record.landmark_id)
            hit5 += int(record.landmark_id in ids)
        denom = max(total, 1)
        return {"samples": total, "recall@1": hit1 / denom, "recall@5": hit5 / denom}

    return (
        _eval(name_features),
        _eval(top_features),
        ego_features,
        ego_records,
        name_features,
    )


def predict_name_id(feature, record, candidate_records, candidate_features):
    valid = [
        index for index, other in enumerate(candidate_records)
        if other.map_name == record.map_name
    ]
    if not valid:
        return None
    sim = feature.unsqueeze(0) @ candidate_features[valid].T
    best = valid[int(sim.argmax(dim=1).item())]
    return candidate_records[best].landmark_id


def evaluate_tracking(top, ego, ego_features, name_features, seed=0):
    if not top or not ego or ego_features is None or name_features is None:
        return {}

    candidate_records = [item[0] for item in top]
    grouped = defaultdict(list)
    for index, (record, _, view, _) in enumerate(ego):
        key = (
            record.map_name,
            record.landmark_id,
            view.get("split", ""),
            view.get("trajectory_index", -1),
        )
        grouped[key].append(index)

    same_total = same_correct = consecutive_total = consecutive_same = 0
    first_prediction = {}
    last_prediction = {}
    sequences = []

    for key, indices in grouped.items():
        indices.sort(key=lambda index: ego[index][1])
        if len(indices) < 2:
            continue
        record = ego[indices[0]][0]
        predictions = [
            predict_name_id(
                ego_features[index],
                record,
                candidate_records,
                name_features,
            )
            for index in indices
        ]
        for prediction in predictions:
            same_total += 1
            same_correct += int(prediction == record.landmark_id)
        for left, right in zip(predictions[:-1], predictions[1:]):
            consecutive_total += 1
            consecutive_same += int(left == right)
        first_prediction[key] = predictions[0]
        last_prediction[key] = predictions[-1]
        sequences.append((key, record, indices, predictions))

    # Build deterministic hard switches. Prefer same-map different landmarks so
    # the metric measures stale target identity instead of trivial map reset.
    rng = random.Random(seed)
    switch_pairs = []
    by_map = defaultdict(list)
    for sequence in sequences:
        by_map[sequence[1].map_name].append(sequence)

    for map_sequences in by_map.values():
        shuffled = list(map_sequences)
        rng.shuffle(shuffled)
        for source in shuffled:
            target = next(
                (
                    candidate for candidate in shuffled
                    if candidate[1].landmark_id != source[1].landmark_id
                ),
                None,
            )
            if target is not None:
                switch_pairs.append((source, target))

    switch_total = switch_correct = stale = 0
    for source, target in switch_pairs:
        source_last_prediction = source[3][-1]
        target_first_prediction = target[3][0]
        switch_total += 1
        switch_correct += int(target_first_prediction == target[1].landmark_id)
        stale += int(
            target_first_prediction == source[1].landmark_id
            and source[1].landmark_id != target[1].landmark_id
        )

    return {
        "same_target_frames": same_total,
        "same_target_accuracy": same_correct / max(same_total, 1),
        "consecutive_pairs": consecutive_total,
        "consecutive_identity_consistency": consecutive_same / max(consecutive_total, 1),
        "scene_switches": switch_total,
        "switch_first_frame_accuracy": switch_correct / max(switch_total, 1),
        "stale_target_rate_after_switch": stale / max(switch_total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        default="../data/cityrefer/landmark_multimodal_map.json",
    )
    parser.add_argument(
        "--split",
        default="val_seen",
        choices=["train_seen", "val_seen", "val_unseen", "test_unseen"],
    )
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument(
        "--siglip-model",
        default="google/siglip-base-patch16-224",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    semantic_map = LandmarkMultimodalMap.load(cache_path)

    print("=== Target understanding: text -> landmark ===")
    print(evaluate_text_target(semantic_map, args.split, args.altitude))

    top, ego = build_visual_index(semantic_map, cache_path)
    print(f"visual assets: top={len(top)} ego={len(ego)}")
    if not top or not ego:
        print(
            "RGB diagnostics skipped: rebuild cache with "
            "--materialize-top-rgb --materialize-ego-rgb"
        )
        return

    encoder = SigLIPEncoder(args.siglip_model, args.device)

    print("=== Name <-> top-view RGB alignment ===")
    print(evaluate_name_to_top(encoder, top))

    print("=== Ego RGB -> landmark identity ===")
    ego_to_name, ego_to_top, ego_features, ego_records, name_features = (
        evaluate_ego_to_name_and_top(encoder, top, ego)
    )
    print("ego->name", ego_to_name)
    print("ego->top_rgb", ego_to_top)

    print("=== Scene/view switch target tracking ===")
    print(
        evaluate_tracking(
            top,
            ego,
            ego_features,
            name_features,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
