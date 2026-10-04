"""Fine-tune the last two BERT and SigLIP vision blocks for static localization.

Starts from the frozen-encoder localizer at the same overhead crop altitude.
Only train_seen labels update weights; val_seen chooses the checkpoint.
"""

import argparse
from collections import defaultdict
import json
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
from multiagent.grounded_instruction_memory import OracleReferenceResolver, SPLITS, load_static_examples
from multiagent.landmark_multimodal_map import LandmarkMultimodalMap
from multiagent.mapdata import GROUND_LEVEL
from multiagent.observation import cropclient
from multiagent.scripts.train_grounded_static_localizer import error_metrics
from multiagent.space import Pose4D


def reconstruct_vocab(examples, feature_cache):
    texts = [""]
    text_ids = {"": 0}
    landmarks = []
    landmark_ids = {}

    def add_text(value):
        value = str(value)
        if value not in text_ids:
            text_ids[value] = len(texts)
            texts.append(value)

    for example in examples:
        memory = example.memory
        add_text(memory.instruction)
        add_text(memory.target_query)
        add_text(memory.attributes)
        for reference in memory.references:
            add_text(reference.record.name)
            key = (memory.map_name, reference.record.landmark_id)
            if key not in landmark_ids:
                landmark_ids[key] = len(landmarks) + 1
                landmarks.append(reference.record)
    if len(texts) != len(feature_cache["text_features"]):
        raise ValueError("Text vocabulary does not match the frozen baseline cache")
    if len(landmarks) != feature_cache["landmark_count"]:
        raise ValueError("Landmark vocabulary does not match the frozen baseline cache")
    return texts, landmarks


def cache_images(landmarks, rgb_dir, altitude, processor, output):
    if output.is_file():
        cached = torch.load(output, map_location="cpu", weights_only=True)
        if cached["altitude"] != altitude or len(cached["pixels"]) != len(landmarks) + 1:
            raise ValueError("Image cache does not match altitude or landmark count")
        return cached["pixels"]
    pixels = torch.zeros((len(landmarks) + 1, 3, 224, 224), dtype=torch.float16)
    cropclient.load_image_cache(image_dir=rgb_dir)
    try:
        for start in range(0, len(landmarks), 32):
            images = []
            for record in landmarks[start:start + 32]:
                pose = Pose4D(record.center_xy[0], record.center_xy[1],
                              GROUND_LEVEL[record.map_name] + altitude, 0.0)
                images.append(Image.fromarray(cropclient.crop_image(
                    record.map_name, pose, (224, 224), "rgb"
                )))
            batch = processor(images=images, return_tensors="pt")["pixel_values"]
            pixels[start + 1:start + 1 + len(images)] = batch.half()
    finally:
        if cropclient._raster_cache is not None:
            for dataset in cropclient._raster_cache.values():
                dataset.close()
        cropclient._raster_cache = cropclient._rgb_cache = cropclient._height_cache = None
    torch.save({"altitude": altitude, "pixels": pixels}, output)
    return pixels


class TunedLocalizer(torch.nn.Module):
    def __init__(self, baseline_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased", local_files_only=True)
        siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224",
                                           local_files_only=True)
        self.vision = siglip.vision_model
        self.head = LandmarkRelativeGoalPredictor()
        self.head.load_state_dict(torch.load(
            baseline_path, map_location="cpu", weights_only=True
        ))
        for param in self.bert.parameters():
            param.requires_grad_(False)
        for param in self.vision.parameters():
            param.requires_grad_(False)
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad_(True)
        for layer in self.vision.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad_(True)
        for block in (self.vision.post_layernorm, self.vision.head):
            for param in block.parameters():
                param.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        # The frozen early BERT blocks must remain deterministic.
        if mode:
            self.bert.embeddings.eval()
            for layer in self.bert.encoder.layer[:-2]:
                layer.eval()
        return self


class BatchBuilder:
    def __init__(self, features, texts, pixels, device):
        self.features = features
        self.pixels = pixels
        self.device = device
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        self.tokenizer = tokenizer
        self.tokens = tokenizer(texts, truncation=True, max_length=192,
                                padding=False)["input_ids"]

    def __call__(self, model, rows):
        features = self.features
        reference_ids = features["reference_indices"][rows]
        text_ids = torch.cat((
            features["instruction_indices"][rows, None],
            features["target_indices"][rows, None],
            features["attribute_indices"][rows, None],
            features["landmark_text_indices"][reference_ids],
        ), dim=1)
        unique_text, inverse_text = torch.unique(text_ids.flatten(), return_inverse=True)
        tokenized = self.tokenizer.pad(
            {"input_ids": [self.tokens[index] for index in unique_text.tolist()]},
            padding=True, return_tensors="pt",
        ).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            encoded_text = model.bert(**tokenized).last_hidden_state[:, 0].float()
        text = encoded_text[inverse_text].reshape(*text_ids.shape, -1)
        unique_images, inverse_images = torch.unique(reference_ids.flatten(),
                                                       return_inverse=True)
        image_batch = self.pixels[unique_images].to(self.device, dtype=torch.float32)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            visual = model.vision(pixel_values=image_batch).pooler_output.float()
        visual = F.normalize(visual, dim=-1)
        visual = visual[inverse_images].reshape(*reference_ids.shape, -1)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            predicted, _, relation_logits = model.head(
                text[:, 0], text[:, 1], text[:, 2], text[:, 3:], visual,
                features["geometry"][reference_ids].to(self.device),
                features["positions"][reference_ids].to(self.device),
                (reference_ids > 0).to(self.device),
            )
        return predicted.float(), relation_logits.float()


@torch.inference_mode()
def evaluate(model, builder, indices, batch_size):
    model.eval()
    predictions = []
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        predicted, _ = builder(model, rows)
        predictions.append(predicted.cpu())
    relative = torch.cat(predictions)
    targets = builder.features["target_relative"][indices]
    errors = ((relative - targets) * 100).norm(dim=-1).numpy()
    world_xy = relative * 100 + builder.features["map_centers"][indices]
    return error_metrics(errors), world_xy.numpy(), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--altitude", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="Diagnostic only: limit batches per epoch")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Check one optimizer step and encoder gradients")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    cityrefer = args.data_root / "cityrefer"
    objects = get_city_refer_objects(
        cityrefer / "objects.json", cityrefer / "processed_descriptions.json"
    )
    semantic_map = LandmarkMultimodalMap.from_cityrefer_objects(objects)
    examples = load_static_examples(args.data_root, OracleReferenceResolver(semantic_map))
    features = torch.load(args.baseline_dir / "feature_cache.pt", map_location="cpu",
                          weights_only=False)
    if len(examples) != 32326 or float(features["top_altitude"]) != args.altitude:
        raise ValueError("Dataset size or overhead altitude does not match baseline")
    if features["episode_ids"] != [example.memory.episode_id for example in examples]:
        raise ValueError("Baseline feature cache does not match episode ordering")
    texts, landmarks = reconstruct_vocab(examples, features)
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224",
                                              local_files_only=True)
    pixels = cache_images(landmarks, args.rgb_dir, args.altitude, processor,
                          args.output_dir / "landmark_pixels.pt")
    builder = BatchBuilder(features, texts, pixels, device)
    model = TunedLocalizer(args.baseline_dir / "text_geometry_visual_best.pt").to(device)
    encoder_params = [param for module in (model.bert, model.vision)
                      for param in module.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": args.head_lr},
        {"params": encoder_params, "lr": args.encoder_lr},
    ], weight_decay=0.01)
    train_indices = features["split_indices"]["train_seen"]
    val_indices = features["split_indices"]["val_seen"]
    if args.smoke_only:
        rows = train_indices[:args.batch_size]
        model.train()
        prediction, relation_logits = builder(model, rows)
        target = features["target_relative"][rows].to(device)
        relations = features["relation_targets"][rows].to(device)
        loss = F.smooth_l1_loss(prediction, target, beta=0.2) + 0.05 * F.binary_cross_entropy_with_logits(relation_logits, relations)
        loss.backward()
        gradients = {
            "bert": float(model.bert.encoder.layer[-1].output.dense.weight.grad.norm()),
            "siglip": float(model.vision.encoder.layers[-1].mlp.fc2.weight.grad.norm()),
        }
        if not torch.isfinite(loss) or not all(np.isfinite(value) and value > 0 for value in gradients.values()):
            raise FloatingPointError(f"Smoke failed: loss={loss}, gradients={gradients}")
        optimizer.step()
        peak_gib = (torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda" else 0.0)
        print(f"SMOKE loss={float(loss.detach()):.5f} gradients={gradients} "
              f"peak_allocated_gib={peak_gib:.2f}", flush=True)
        return
    baseline_metrics, _, _ = evaluate(model, builder, val_indices, args.batch_size)
    print(f"before fine-tuning val_seen={baseline_metrics}", flush=True)
    history = []
    best_median = baseline_metrics["median_error_m"]
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        shuffled = train_indices[torch.randperm(len(train_indices))]
        total_loss = 0.0
        samples = 0
        for batch_number, start in enumerate(range(0, len(shuffled), args.batch_size)):
            if args.max_train_batches and batch_number >= args.max_train_batches:
                break
            rows = shuffled[start:start + args.batch_size]
            prediction, relation_logits = builder(model, rows)
            target = features["target_relative"][rows].to(device)
            relations = features["relation_targets"][rows].to(device)
            location_loss = F.smooth_l1_loss(prediction, target, beta=0.2)
            relation_loss = F.binary_cross_entropy_with_logits(relation_logits, relations)
            loss = location_loss + 0.05 * relation_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_number}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
            samples += len(rows)
            if (batch_number + 1) % 200 == 0:
                print(f"epoch={epoch} batch={batch_number + 1} samples={samples}", flush=True)
        validation, _, _ = evaluate(model, builder, val_indices, args.batch_size)
        entry = {"epoch": epoch, "train_samples": samples,
                 "train_loss": total_loss / samples,
                 "seconds": time.time() - start_time, "val_seen": validation}
        history.append(entry)
        print(f"epoch={epoch} {entry}", flush=True)
        if validation["median_error_m"] < best_median:
            best_median = validation["median_error_m"]
            best_epoch = epoch
            torch.save({"head": model.head.state_dict(),
                        "bert": {key: tensor for key, tensor in model.bert.state_dict().items()
                                 if key.startswith("encoder.layer.10.") or
                                 key.startswith("encoder.layer.11.")},
                        "vision": {key: tensor for key, tensor in model.vision.state_dict().items()
                                   if key.startswith("encoder.layers.10.") or
                                   key.startswith("encoder.layers.11.") or
                                   key.startswith("post_layernorm.") or
                                   key.startswith("head.")}},
                       args.output_dir / "best_finetuned.pt")
    if best_epoch:
        saved = torch.load(args.output_dir / "best_finetuned.pt", map_location="cpu",
                           weights_only=True)
        model.head.load_state_dict(saved["head"])
        model.bert.load_state_dict(saved["bert"], strict=False)
        model.vision.load_state_dict(saved["vision"], strict=False)
    report = {"scope": "32,326 static rows", "altitude_m": args.altitude,
              "encoder_finetune": "last two BERT layers; last two SigLIP vision layers + pooling",
              "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
              "batch_size": args.batch_size, "epochs": args.epochs,
              "baseline_val_seen": baseline_metrics, "best_epoch": best_epoch,
              "history": history, "splits": {}}
    with (args.output_dir / "static_predictions.jsonl").open("w") as output:
        for split in SPLITS:
            indices = features["split_indices"][split]
            metrics, positions, errors = evaluate(model, builder, indices, args.batch_size)
            report["splits"][split] = metrics
            for local, row in enumerate(indices.tolist()):
                output.write(json.dumps({"split": split,
                                         "source_row": features["source_rows"][row],
                                         "episode_id": features["episode_ids"][row],
                                         "predicted_xy": positions[local].tolist(),
                                         "error_m": float(errors[local])}) + "\n")
            print(f"FINAL {split}={metrics}", flush=True)
    (args.output_dir / "finetuning_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
