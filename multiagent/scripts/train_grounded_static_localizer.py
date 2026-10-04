"""Train and evaluate one-shot landmark-relative localization on all CityNav splits.

The only learned labels come from train_seen. val_seen selects the checkpoint;
val_unseen and test_unseen are evaluated only after selection. No Stage-2,
student rollout, target-object identity token, or target coordinate enters the
compiled instruction memory.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from multiagent.cityreferobject import get_city_refer_objects
from multiagent.grounded_goal_predictor import LandmarkRelativeGoalPredictor
from multiagent.grounded_instruction_memory import (
    OracleReferenceResolver, SPLITS, load_static_examples, map_center, polygon_area,
)
from multiagent.landmark_multimodal_map import LandmarkMultimodalMap
from multiagent.mapdata import GROUND_LEVEL
from multiagent.observation import cropclient
from multiagent.space import Pose4D


def encode_texts(texts, device, batch_size=64):
    """Encode each distinct string once using frozen BERT (up to 192 tokens)."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    model = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True)
    model.to(device).eval()
    encoded = torch.zeros((len(texts), model.config.hidden_size), dtype=torch.float16)
    truncation_count = 0
    start_time = time.time()
    with torch.inference_mode():
        for start in range(1, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokenized = tokenizer(
                batch, padding=True, truncation=True, max_length=192,
                return_tensors="pt", return_overflowing_tokens=False,
            )
            # A separate length check makes any lost instruction content visible.
            lengths = tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
            truncation_count += sum(len(ids) > 192 for ids in lengths)
            tokenized = tokenized.to(device)
            hidden = model(**tokenized).last_hidden_state[:, 0, :]
            encoded[start:start + len(batch)] = hidden.float().cpu().to(torch.float16)
            if start % (batch_size * 100) == 1:
                print(f"BERT texts {min(start + len(batch), len(texts) - 1)}/{len(texts) - 1}", flush=True)
    print(f"BERT encoded {len(texts) - 1} strings in {time.time() - start_time:.1f}s; "
          f"truncated={truncation_count}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return encoded, truncation_count


def encode_visuals(records, rgb_dir, device, batch_size=24, top_altitude=80.0):
    """One overhead crop per referenced entity, never at the target location."""
    processor = AutoProcessor.from_pretrained(
        "google/siglip-base-patch16-224", local_files_only=True
    )
    model = AutoModel.from_pretrained(
        "google/siglip-base-patch16-224", local_files_only=True
    ).to(device).eval()
    dimension = int(model.config.vision_config.hidden_size)
    encoded = torch.zeros((len(records) + 1, dimension), dtype=torch.float16)
    cropclient.load_image_cache(image_dir=rgb_dir)
    start_time = time.time()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            images = []
            for record in batch:
                pose = Pose4D(
                    record.center_xy[0], record.center_xy[1],
                    GROUND_LEVEL[record.map_name] + top_altitude, 0.0,
                )
                rgb = cropclient.crop_image(record.map_name, pose, (224, 224), "rgb")
                images.append(Image.fromarray(rgb))
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
            features = F.normalize(features.float(), dim=-1)
            encoded[start + 1:start + len(batch) + 1] = features.cpu().to(torch.float16)
            if start % (batch_size * 20) == 0:
                print(f"SigLIP landmark crops {start + len(batch)}/{len(records)}", flush=True)
    print(f"SigLIP encoded {len(records)} landmark crops in "
          f"{time.time() - start_time:.1f}s", flush=True)
    # cropclient.clear_image_cache() currently iterates raster-cache keys rather
    # than dataset handles. Close the handles here until that shared helper is fixed.
    if cropclient._raster_cache is not None:
        for dataset in cropclient._raster_cache.values():
            dataset.close()
    cropclient._raster_cache = cropclient._rgb_cache = cropclient._height_cache = None
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return encoded


def build_feature_cache(examples, rgb_dir, device, text_batch_size, visual_batch_size,
                        top_altitude=80.0):
    text_ids = {"": 0}
    texts = [""]
    landmark_ids = {}
    landmarks = []

    def text_id(value):
        value = str(value)
        if value not in text_ids:
            text_ids[value] = len(texts)
            texts.append(value)
        return text_ids[value]

    for example in examples:
        memory = example.memory
        text_id(memory.instruction)
        text_id(memory.target_query)
        text_id(memory.attributes)
        for reference in memory.references:
            text_id(reference.record.name)
            key = (memory.map_name, reference.record.landmark_id)
            if key not in landmark_ids:
                landmark_ids[key] = len(landmarks) + 1
                landmarks.append(reference.record)

    text_features, truncated = encode_texts(texts, device, text_batch_size)
    visual_features = encode_visuals(
        landmarks, rgb_dir, device, visual_batch_size, top_altitude
    )
    geometry = torch.zeros((len(landmarks) + 1, 11), dtype=torch.float32)
    positions = torch.zeros((len(landmarks) + 1, 2), dtype=torch.float32)
    landmark_text_indices = torch.zeros(len(landmarks) + 1, dtype=torch.long)
    object_types = ("Building", "TrafficRoad", "Parking")
    for index, record in enumerate(landmarks, 1):
        center = map_center(record.map_name)
        relative = ((record.center_xy[0] - center[0]) / 100,
                    (record.center_xy[1] - center[1]) / 100)
        points = record.polygon or [record.center_xy]
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        width = (max(x_values) - min(x_values)) / 100
        height = (max(y_values) - min(y_values)) / 100
        size = math.sqrt(polygon_area(record.polygon)) / 100
        type_code = [float(record.object_type == name) for name in object_types]
        type_code.append(float(record.object_type not in object_types))
        geometry[index] = torch.tensor([
            relative[0], relative[1], record.normalized_xy[0], record.normalized_xy[1],
            width, height, size, *type_code,
        ])
        positions[index] = torch.tensor(relative)
        landmark_text_indices[index] = text_id(record.name)

    max_references = max(len(example.memory.references) for example in examples)
    rows = len(examples)
    reference_indices = torch.zeros((rows, max_references), dtype=torch.long)
    instruction_indices = torch.zeros(rows, dtype=torch.long)
    target_indices = torch.zeros(rows, dtype=torch.long)
    attribute_indices = torch.zeros(rows, dtype=torch.long)
    map_centers = torch.zeros((rows, 2), dtype=torch.float32)
    target_relative = torch.zeros((rows, 2), dtype=torch.float32)
    relation_targets = torch.zeros((rows, 10), dtype=torch.float32)
    split_indices = defaultdict(list)
    for index, example in enumerate(examples):
        memory = example.memory
        instruction_indices[index] = text_ids[memory.instruction]
        target_indices[index] = text_ids[memory.target_query]
        attribute_indices[index] = text_ids[memory.attributes]
        for slot, reference in enumerate(memory.references):
            reference_indices[index, slot] = landmark_ids[(
                memory.map_name, reference.record.landmark_id
            )]
        center = map_center(memory.map_name)
        map_centers[index] = torch.tensor(center)
        target_relative[index] = (torch.tensor(example.target_xy) - map_centers[index]) / 100
        relation_targets[index] = torch.tensor(memory.relation_labels)
        split_indices[example.split].append(index)

    return {
        "text_features": text_features,
        "visual_features": visual_features,
        "landmark_text_indices": landmark_text_indices,
        "geometry": geometry,
        "positions": positions,
        "reference_indices": reference_indices,
        "instruction_indices": instruction_indices,
        "target_indices": target_indices,
        "attribute_indices": attribute_indices,
        "map_centers": map_centers,
        "target_relative": target_relative,
        "relation_targets": relation_targets,
        "split_indices": {name: torch.tensor(indices, dtype=torch.long)
                          for name, indices in split_indices.items()},
        "episode_ids": [example.memory.episode_id for example in examples],
        "source_rows": [example.source_row for example in examples],
        "resolved_counts": [len(example.memory.references) for example in examples],
        "text_truncated": truncated,
        "landmark_count": len(landmarks),
        "top_altitude": float(top_altitude),
    }


def model_batch(features, indices, device):
    reference_ids = features["reference_indices"][indices]
    text = features["text_features"]
    landmark_text = features["landmark_text_indices"][reference_ids]
    return (
        text[features["instruction_indices"][indices]].float().to(device),
        text[features["target_indices"][indices]].float().to(device),
        text[features["attribute_indices"][indices]].float().to(device),
        text[landmark_text].float().to(device),
        features["visual_features"][reference_ids].float().to(device),
        features["geometry"][reference_ids].to(device),
        features["positions"][reference_ids].to(device),
        (reference_ids > 0).to(device),
    )


def error_metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "samples": int(errors.size),
        "hit@10m": float(np.mean(errors <= 10)),
        "hit@20m": float(np.mean(errors <= 20)),
        "hit@30m": float(np.mean(errors <= 30)),
        "median_error_m": float(np.median(errors)),
        "mean_error_m": float(np.mean(errors)),
    }


@torch.inference_mode()
def evaluate(model, features, indices, device, batch_size, return_predictions=False):
    model.eval()
    predictions = []
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        predicted, _, _ = model(*model_batch(features, rows, device))
        predictions.append(predicted.cpu())
    predicted_relative = torch.cat(predictions)
    targets = features["target_relative"][indices]
    errors = ((predicted_relative - targets) * 100).norm(dim=-1).numpy()
    result = error_metrics(errors)
    if return_predictions:
        world_xy = predicted_relative * 100 + features["map_centers"][indices]
        return result, world_xy.numpy(), errors
    return result


def train_variant(features, device, args, use_visual):
    torch.manual_seed(args.seed)
    model = LandmarkRelativeGoalPredictor(
        text_dim=features["text_features"].shape[1],
        visual_dim=features["visual_features"].shape[1],
        use_visual=use_visual,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_indices = features["split_indices"]["train_seen"]
    validation_indices = features["split_indices"]["val_seen"]
    history = []
    best_median = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        shuffled = train_indices[torch.randperm(len(train_indices))]
        training_loss = 0.0
        start_time = time.time()
        for start in range(0, len(shuffled), args.batch_size):
            indices = shuffled[start:start + args.batch_size]
            predicted, _, relation_logits = model(*model_batch(features, indices, device))
            target = features["target_relative"][indices].to(device)
            relations = features["relation_targets"][indices].to(device)
            location_loss = F.smooth_l1_loss(predicted, target, beta=0.2)
            relation_loss = F.binary_cross_entropy_with_logits(relation_logits, relations)
            loss = location_loss + 0.05 * relation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_loss += float(loss.item()) * len(indices)
        validation = evaluate(model, features, validation_indices, device, args.batch_size)
        seconds = time.time() - start_time
        entry = {
            "epoch": epoch + 1,
            "train_loss": training_loss / len(train_indices),
            "seconds": seconds,
            "val_seen": validation,
        }
        history.append(entry)
        print(f"variant={'visual' if use_visual else 'text_geometry'} "
              f"epoch={epoch + 1} time={seconds:.1f}s "
              f"train_loss={entry['train_loss']:.4f} "
              f"val_hit20={validation['hit@20m']:.4f} "
              f"val_median={validation['median_error_m']:.2f}m", flush=True)
        if validation["median_error_m"] < best_median:
            best_median = validation["median_error_m"]
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--visual-batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--top-altitude", type=float, default=30.0)
    parser.add_argument("--only-visual", action="store_true",
                        help="Skip the unchanged text+geometry ablation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cityrefer = args.data_root / "cityrefer"
    objects = get_city_refer_objects(
        cityrefer / "objects.json", cityrefer / "processed_descriptions.json"
    )
    semantic_map = LandmarkMultimodalMap.from_cityrefer_objects(objects)
    resolver = OracleReferenceResolver(semantic_map)
    examples = load_static_examples(args.data_root, resolver)
    if len(examples) != 32326:
        raise ValueError(f"Expected all 32,326 source rows; got {len(examples)}")
    feature_path = args.output_dir / "feature_cache.pt"
    if feature_path.is_file():
        features = torch.load(feature_path, map_location="cpu", weights_only=False)
        if float(features.get("top_altitude", 80.0)) != args.top_altitude:
            raise ValueError("Cached visual features use a different top altitude")
        print(f"Loaded cached features: {feature_path}", flush=True)
    else:
        features = build_feature_cache(
            examples, args.rgb_dir, device, args.text_batch_size,
            args.visual_batch_size, args.top_altitude,
        )
        torch.save(features, feature_path)
        print(f"Saved cached features: {feature_path}", flush=True)
    if any(len(features["split_indices"][split]) != sum(
        example.split == split for example in examples
    ) for split in SPLITS):
        raise ValueError("Cached features do not cover the current dataset splits")
    if features["episode_ids"] != [example.memory.episode_id for example in examples]:
        raise ValueError("Cached features do not match the current episode ordering")

    report = {
        "scope": "32,326 refined CityNav source rows; train_seen labels only for learning",
        "checkpoint_selection": "lowest val_seen median error; unseen splits untouched until final evaluation",
        "oracle_inputs": "processed reference names, canonical instruction and static map records; target IDs excluded from references",
        "visual_input": "one frozen SigLIP overhead RGB crop centered at each referenced landmark",
        "text_input": "one frozen BERT embedding per unique instruction/target phrase/attribute/landmark name",
        "config": {
            "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "top_altitude_m": args.top_altitude,
            "text_truncated": features["text_truncated"],
            "unique_visual_landmarks": features["landmark_count"],
        },
        "variants": {},
    }
    prediction_path = args.output_dir / "static_predictions.jsonl"
    with prediction_path.open("w") as prediction_file:
        variants = (("text_geometry_visual", True),) if args.only_visual else (
            ("text_geometry", False), ("text_geometry_visual", True)
        )
        for variant, use_visual in variants:
            model, history = train_variant(features, device, args, use_visual)
            torch.save(model.state_dict(), args.output_dir / f"{variant}_best.pt")
            result = {"history": history, "splits": {}, "subgroups": {}}
            for split in SPLITS:
                indices = features["split_indices"][split]
                split_metrics, world_xy, errors = evaluate(
                    model, features, indices, device, args.batch_size,
                    return_predictions=True,
                )
                result["splits"][split] = split_metrics
                subgroup_errors = defaultdict(list)
                for local_index, source_index in enumerate(indices.tolist()):
                    count = features["resolved_counts"][source_index]
                    group = "0" if count == 0 else "1" if count == 1 else "2" if count == 2 else "3+"
                    subgroup_errors[group].append(float(errors[local_index]))
                    prediction_file.write(json.dumps({
                        "variant": variant,
                        "split": split,
                        "source_row": features["source_rows"][source_index],
                        "episode_id": features["episode_ids"][source_index],
                        "resolved_landmarks": count,
                        "predicted_xy": world_xy[local_index].tolist(),
                        "error_m": float(errors[local_index]),
                    }) + "\n")
                result["subgroups"][split] = {
                    group: error_metrics(group_values)
                    for group, group_values in subgroup_errors.items()
                }
                print(f"FINAL variant={variant} split={split} {split_metrics}", flush=True)
            report["variants"][variant] = result
    (args.output_dir / "static_localization_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"Complete: {args.output_dir / 'static_localization_report.json'}", flush=True)


if __name__ == "__main__":
    main()
