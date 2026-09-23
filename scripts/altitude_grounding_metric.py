"""Mask and box metrics with per-sample output for altitude experiments."""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from pycocotools import mask as mask_utils


def foreground_box(binary):
    ys, xs = torch.where(binary)
    if xs.numel() == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def box_iou(first, second):
    if first is None or second is None:
        return 0.0
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left + 1) * max(0, bottom - top + 1)
    first_area = (first[2] - first[0] + 1) * (first[3] - first[1] + 1)
    second_area = (second[2] - second[0] + 1) * (second[3] - second[1] + 1)
    return intersection / (first_area + second_area - intersection)


def encode_mask(binary):
    encoded = mask_utils.encode(np.asfortranarray(binary.cpu().numpy().astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


@METRICS.register_module()
class AltitudeGroundingMetric(BaseMetric):
    default_prefix = "AltitudeGrounding"

    def __init__(self, collect_device="cpu", prefix=None, output_dir=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.output_dir = output_dir or os.environ.get("ALTITUDE_METRIC_DIR")

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            prediction = sample["pred_sem_seg"]["data"].squeeze().cpu() == 1
            target = sample["gt_sem_seg"]["data"].squeeze().cpu() == 1
            pred_box, target_box = foreground_box(prediction), foreground_box(target)
            intersection = torch.logical_and(prediction, target).sum().item()
            union = torch.logical_or(prediction, target).sum().item()
            current_box_iou = box_iou(pred_box, target_box)
            center_hit = False
            if pred_box is not None and target_box is not None:
                center_x = (pred_box[0] + pred_box[2]) / 2
                center_y = (pred_box[1] + pred_box[3]) / 2
                center_hit = (
                    target_box[0] <= center_x <= target_box[2]
                    and target_box[1] <= center_y <= target_box[3]
                )
            self.results.append({
                "img_path": sample.get("img_path", ""),
                "text": sample.get("text", ""),
                "group": sample.get("category_name", "unknown"),
                "mask_iou": intersection / union if union else 0.0,
                "box_iou": current_box_iou,
                "pred_box": pred_box,
                "gt_box": target_box,
                "center_hit": bool(center_hit),
                "nonempty": pred_box is not None,
                "pred_rle": encode_mask(prediction),
            })

    @staticmethod
    def summarize(rows):
        if not rows:
            return {"count": 0}
        mask_values = np.asarray([item["mask_iou"] for item in rows])
        box_values = np.asarray([item["box_iou"] for item in rows])
        return {
            "count": len(rows),
            "mask_gIoU": float(mask_values.mean() * 100),
            "mask_Pr@0.5": float((mask_values > 0.5).mean() * 100),
            "mean_box_iou": float(box_values.mean() * 100),
            "box_acc_0.25": float((box_values >= 0.25).mean() * 100),
            "box_acc_0.5": float((box_values >= 0.5).mean() * 100),
            "center_hit_rate": float(np.mean([item["center_hit"] for item in rows]) * 100),
            "nonempty_prediction_rate": float(np.mean([item["nonempty"] for item in rows]) * 100),
        }

    def compute_metrics(self, results):
        grouped = defaultdict(list)
        for row in results:
            grouped[row["group"]].append(row)
        summary = {
            "overall": self.summarize(results),
            "groups": {key: self.summarize(value) for key, value in sorted(grouped.items())},
        }
        if self.output_dir:
            output = Path(self.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            with (output / "per_sample_metrics.jsonl").open("w") as stream:
                for row in results:
                    stream.write(json.dumps(row) + "\n")
            (output / "grounding_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        flattened = dict(summary["overall"])
        for group, values in summary["groups"].items():
            flattened[f"{group}/count"] = values["count"]
            flattened[f"{group}/mean_box_iou"] = values["mean_box_iou"]
            flattened[f"{group}/box_acc_0.25"] = values["box_acc_0.25"]
        return flattened
