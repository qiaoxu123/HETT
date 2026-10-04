#!/usr/bin/env python3
"""Apply the prior partial-tuned SigLIP2 head to real teacher-pose RGB."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from multiagent.models.multi_attribute_head import MultiAttributeHead  # noqa: E402
from multiagent.scene_grounding.scene_evidence import combine_probabilities  # noqa: E402


def read_rgb(root, relative):
    bgr = cv2.imread(str(root / relative))
    if bgr is None: raise FileNotFoundError(root / relative)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="google/siglip2-base-patch16-256")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if "test_unseen" in " ".join(map(str, vars(args).values())): raise ValueError("test_unseen is forbidden")
    from transformers import AutoImageProcessor, AutoModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(saved["model"]); encoder.to(device).eval().requires_grad_(False)
    classes = saved["classes"]
    head = MultiAttributeHead(encoder.config.vision_config.hidden_size, {key: len(value) for key, value in classes.items()})
    head.load_state_dict(saved["head"]); head.to(device).eval().requires_grad_(False)
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines()]
    jobs = []
    for row_index, row in enumerate(rows):
        jobs.append((row_index, "whole", -1, row["image"]))
        jobs.extend((row_index, "region", region_index, region["image"])
                    for region_index, region in enumerate(row["regions"]))
    predictions = [{"whole": None, "regions": [None] * len(row["regions"])} for row in rows]
    started = time.time()
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start:start + args.batch_size]
        images = [read_rgb(args.dataset, item[3]) for item in batch]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = head(encoder.get_image_features(pixel_values=pixels))
        probabilities = {name: torch.softmax(value.float(), -1).cpu().numpy() for name, value in output.items()}
        for local, (row_index, kind, region_index, _) in enumerate(batch):
            value = {name: {label: float(probabilities[name][local, label_index])
                            for label_index, label in enumerate(labels)} for name, labels in classes.items()}
            if kind == "whole": predictions[row_index]["whole"] = value
            else: predictions[row_index]["regions"][region_index] = value
        if start and start % (args.batch_size * 100) == 0: print(start, len(jobs), flush=True)
    output_rows = []
    for row, prediction in zip(rows, predictions):
        evidence = combine_probabilities(prediction["whole"], prediction["regions"])
        image = read_rgb(args.dataset, row["image"])
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        confidence = float(np.mean([max(values.values()) for values in evidence.combined.values()]))
        output_rows.append({
            "sample_id": row["sample_id"], "episode_id": row["episode_id"], "split": row["split"],
            "map_name": row["map_name"], "step": row["step"], "distance_m": row["goal_distance_m"],
            "altitude_m": row["altitude_m"], "template_id": row["template_id"],
            "whole": evidence.whole, "regions": evidence.regions, "combined": evidence.combined,
            "region_count": evidence.region_count,
            "view_quality": {"attribute_confidence": confidence,
                             "blur_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                             "region_count": evidence.region_count,
                             "largest_region_fraction": max((r["visible_area_m2"] for r in row["regions"]), default=0.0) /
                                                        max((row["altitude_m"] * 2) ** 2, 1.0)},
        })
    with (args.output / "evidence.jsonl").open("w") as stream:
        for row in output_rows: stream.write(json.dumps(row) + "\n")
    metrics = {"frames": len(rows), "image_jobs": len(jobs), "seconds": time.time() - started,
               "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
               "model_name": args.model_name, "checkpoint": str(args.checkpoint),
               "observation_source": "teacher_trajectory_pose", "test_unseen_read": False}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
