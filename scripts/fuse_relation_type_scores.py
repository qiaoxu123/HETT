#!/usr/bin/env python3
"""Fuse accepted relation scores with independently predicted 40 m type scores."""

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_multiscale_type_matching import encode_candidate_crops
from scripts.audit_target_type_alignment import first_type_match, train_probe
from scripts.train_relation_geometry_ranker import (
    CUE_NAMES,
    RelationGeometryRanker,
    build_split,
    load_rows,
    matched_same_map_switch_indices,
    scores as relation_scores,
)
from scripts.train_rgb_candidate_ranker import metrics, prepare_rows


SPLITS = ("val_seen", "val_unseen")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-dir", type=Path,
        default=Path("runs/candidate_rgb_data/geometry_top5_oof_grouped_enriched"))
    parser.add_argument(
        "--relation-dir", type=Path,
        default=Path("runs/relation_geometry_ranker_contrastive"))
    parser.add_argument(
        "--type-cache", type=Path,
        default=Path("runs/target_type_alignment/features.pt"))
    parser.add_argument(
        "--scale40-cache", type=Path,
        default=Path("runs/target_type_alignment/multiscale/scale_40.pt"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/relation_type_fusion"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument(
        "--hf-cache", type=Path,
        default=Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--seeds", default="0,1,2")
    return parser.parse_args()


def row_key(row):
    return (row["map_name"], row["description"],
            tuple(round(float(value), 7) for value in row["target"]))


def remap_features(source_rows, source_features, target_rows):
    """Reorder a feature tensor using stable semantic keys and occurrence order."""
    lookup = defaultdict(deque)
    for index, row in enumerate(source_rows):
        lookup[row_key(row)].append(index)
    indices = []
    for row in target_rows:
        key = row_key(row)
        if not lookup[key]:
            raise ValueError(f"feature row is missing: {key[:2]}")
        indices.append(lookup[key].popleft())
    if any(values for values in lookup.values()):
        raise ValueError("unused feature rows remain after alignment")
    return source_features[torch.tensor(indices)]


def validate_text_alignment(rows, data_root, split):
    trajectories = json.load(open(
        data_root / "processed_citynav" / f"citynav_{split}.json"))
    descriptions = [row["descriptions"][0] for row in trajectories
                    if row.get("dist_marker_to_target", 0.0) <= 30.0]
    if descriptions != [row["description"] for row in rows]:
        raise ValueError(f"text cache order mismatch for {split}")


def load_or_encode_candidates(args, rows, device):
    output = args.output_dir / "scale40_candidate_features.pt"
    if output.exists():
        cached = torch.load(output, map_location="cpu", weights_only=True)
        if all(cached[split].shape[:2] == (len(rows[split]), 5) for split in SPLITS):
            return cached
        raise ValueError("cached 40 m candidate features have the wrong shape")

    scale40 = torch.load(args.scale40_cache, map_location="cpu", weights_only=True)
    old_unseen, _ = prepare_rows(
        args.candidate_dir / "val_unseen.jsonl", len(rows["val_unseen"]), 0, 5, False)
    unseen = remap_features(old_unseen, scale40["candidate"], rows["val_unseen"])

    processor = AutoProcessor.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    print(f"encoding 40 m val_seen candidates: {len(rows['val_seen']) * 5}", flush=True)
    seen = encode_candidate_crops(
        rows["val_seen"], encoder, processor, args.data_root, device,
        40.0, args.feature_batch_size)
    cached = {"val_seen": seen, "val_unseen": unseen}
    torch.save(cached, output)
    del encoder
    torch.cuda.empty_cache()
    return cached


def normalized(score):
    centered = score - score.mean(-1, keepdim=True)
    return centered / centered.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)


@torch.inference_mode()
def type_compatibility(text_model, image_model, text_features, image_features, device):
    text_logits = text_model(text_features.float().to(device)).cpu()
    shape = image_features.shape
    image_logits = image_model(
        image_features.float().reshape(-1, shape[-1]).to(device)
    ).cpu().reshape(shape[0], shape[1], -1)
    log_text = F.log_softmax(text_logits, dim=-1)[:, None]
    log_image = F.log_softmax(image_logits, dim=-1)
    compatibility = torch.logsumexp(log_text + log_image, dim=-1)
    return compatibility, text_logits, image_logits


def fused_rank(relation, type_score, prior, weights, landmark_count):
    score = torch.empty_like(relation)
    groups = (landmark_count.eq(1), landmark_count.ne(1))
    for name, selected in zip(("one", "multi"), groups):
        rw, tw, pw = weights[name]
        score[selected] = (rw * relation[selected] + tw * type_score[selected]
                           + pw * prior[selected])
    return score.argmax(-1)


def tune_weights(relation, type_score, distance, landmark_count):
    prior = normalized(-torch.arange(relation.shape[1], dtype=relation.dtype)[None].expand_as(
        relation))
    grids = {
        "relation": (0.0, 0.25, 0.5, 1.0, 2.0),
        "type": (0.0, 0.5, 1.0, 2.0),
        "prior": (0.0, 0.1, 0.25, 0.5, 1.0),
    }
    selected_weights = {}
    for name, selected in (("one", landmark_count.eq(1)),
                           ("multi", landmark_count.ne(1))):
        best = None
        for rw in grids["relation"]:
            for tw in grids["type"]:
                for pw in grids["prior"]:
                    if rw + tw + pw == 0:
                        continue
                    rank = (rw * relation[selected] + tw * type_score[selected]
                            + pw * prior[selected]).argmax(-1)
                    value = metrics(distance[selected], rank)
                    key = value["hit20"], -value["mean_ne"], -sum((rw, tw, pw))
                    if best is None or key > best[0]:
                        best = key, (rw, tw, pw)
        selected_weights[name] = best[1]
    return selected_weights


def by_landmark_count(distance, rank, count):
    result = {}
    for name, selected in (("one", count.eq(1)), ("two", count.eq(2)),
                           ("three_plus", count.ge(3))):
        result[name] = {"count": int(selected.sum()), **metrics(distance[selected], rank[selected])}
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {split: load_rows(args.candidate_dir / f"{split}.jsonl", 5)
            for split in SPLITS}
    for split in SPLITS:
        validate_text_alignment(rows[split], args.data_root, split)

    relation_features = torch.load(
        args.relation_dir / "relation_features.pt", map_location="cpu", weights_only=True)
    relation_data = {
        split: build_split(rows[split], relation_features[split], 9) for split in SPLITS}
    type_features = torch.load(args.type_cache, map_location="cpu", weights_only=True)
    scale40 = torch.load(args.scale40_cache, map_location="cpu", weights_only=True)
    candidate_image = load_or_encode_candidates(args, rows, device)

    seeds = [int(value) for value in args.seeds.split(",") if value]
    result = {
        "config": {
            "seeds": seeds,
            "selection": "two group score weights selected only on val_seen",
            "groups": "one landmark versus all other landmark counts",
            "score_normalization": "per-row RMS after mean centering",
        },
        "anti_leakage": {
            "target_coordinates": "weight selection metrics on val_seen; final metrics on val_unseen",
            "val_unseen_used_for_weight_selection": False,
            "inference_inputs": "relation/landmark geometry plus predicted text/RGB target types",
        },
        "runs": [],
    }
    prior = {
        split: normalized(-torch.arange(5, dtype=torch.float32)[None].expand(
            len(rows[split]), 5)) for split in SPLITS}
    for seed in seeds:
        text_model = train_probe(
            type_features["text"]["train_seen"], type_features["text"]["val_seen"],
            seed, 40, device)
        image_model = train_probe(
            scale40["target"]["train_seen"], scale40["target"]["val_seen"],
            seed, 40, device)
        relation_model = RelationGeometryRanker(cue_dim=len(CUE_NAMES)).to(device)
        relation_model.load_state_dict(torch.load(
            args.relation_dir / f"model_seed{seed}.pt", map_location="cpu",
            weights_only=True))
        relation_model.eval()

        split_scores = {}
        for split in SPLITS:
            relation = normalized(relation_scores(
                relation_model, relation_data[split], device))
            type_score, text_logits, image_logits = type_compatibility(
                text_model, image_model, type_features["text"][split]["features"],
                candidate_image[split], device)
            split_scores[split] = {
                "relation": relation,
                "type": normalized(type_score),
                "text_logits": text_logits,
                "image_logits": image_logits,
            }
        weights = tune_weights(
            split_scores["val_seen"]["relation"], split_scores["val_seen"]["type"],
            relation_data["val_seen"]["distance"],
            relation_data["val_seen"]["landmark_count"])

        unseen = relation_data["val_unseen"]
        normal_rank = fused_rank(
            split_scores["val_unseen"]["relation"], split_scores["val_unseen"]["type"],
            prior["val_unseen"], weights, unseen["landmark_count"])
        wrong_relation = normalized(relation_scores(
            relation_model, unseen, device, "wrong_relation"))
        wrong_landmark = normalized(relation_scores(
            relation_model, unseen, device, "wrong_landmark"))
        switched = matched_same_map_switch_indices(rows["val_unseen"])
        wrong_type_raw = torch.logsumexp(
            F.log_softmax(split_scores["val_unseen"]["text_logits"][switched], -1)[:, None]
            + F.log_softmax(split_scores["val_unseen"]["image_logits"], -1), -1)
        wrong_type = normalized(wrong_type_raw)
        permuted_type_raw = torch.logsumexp(
            F.log_softmax(split_scores["val_unseen"]["text_logits"].roll(1, -1), -1)[:, None]
            + F.log_softmax(split_scores["val_unseen"]["image_logits"], -1), -1)
        permuted_type = normalized(permuted_type_raw)
        distance = unseen["distance"]
        relation_only = split_scores["val_unseen"]["relation"].argmax(-1)
        text_prediction = split_scores["val_unseen"]["text_logits"].argmax(-1)
        image_prediction = split_scores["val_unseen"]["image_logits"].argmax(-1)
        hard_type, _ = first_type_match(text_prediction, image_prediction)
        run = {
            "seed": seed,
            "selected_weights": {name: list(values) for name, values in weights.items()},
            "val_unseen": {
                "fusion": metrics(distance, normal_rank),
                "wrong_relation": metrics(distance, fused_rank(
                    wrong_relation, split_scores["val_unseen"]["type"],
                    prior["val_unseen"], weights, unseen["landmark_count"])),
                "wrong_landmark": metrics(distance, fused_rank(
                    wrong_landmark, split_scores["val_unseen"]["type"],
                    prior["val_unseen"], weights, unseen["landmark_count"])),
                "wrong_type": metrics(distance, fused_rank(
                    split_scores["val_unseen"]["relation"], wrong_type,
                    prior["val_unseen"], weights, unseen["landmark_count"])),
                "permuted_type_classes": metrics(distance, fused_rank(
                    split_scores["val_unseen"]["relation"], permuted_type,
                    prior["val_unseen"], weights, unseen["landmark_count"])),
                "relation_only": metrics(distance, relation_only),
                "hard_type_match": metrics(distance, hard_type),
                "candidate_prior": metrics(distance, torch.zeros(len(distance), dtype=torch.long)),
                "oracle_top5": metrics(distance, distance.argmin(-1)),
                "by_landmark_count": by_landmark_count(
                    distance, normal_rank, unseen["landmark_count"]),
            },
        }
        result["runs"].append(run)
        del text_model, image_model, relation_model
        torch.cuda.empty_cache()

    keys = ("fusion", "wrong_relation", "wrong_landmark", "wrong_type",
            "permuted_type_classes", "relation_only", "hard_type_match",
            "candidate_prior", "oracle_top5")
    result["val_unseen_mean"] = {
        key: {metric: sum(run["val_unseen"][key][metric] for run in result["runs"])
              / len(result["runs"])
              for metric in ("mean_ne", "median_ne", "hit20", "hit50")}
        for key in keys
    }
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["val_unseen_mean"], indent=2))


if __name__ == "__main__":
    main()
