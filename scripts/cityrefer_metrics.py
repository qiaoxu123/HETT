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


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left + 1) * max(0, bottom - top + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return intersection / (area_a + area_b - intersection)


def encode_mask(binary):
    rle = mask_utils.encode(np.asfortranarray(binary.cpu().numpy().astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


@METRICS.register_module()
class CityReferGroundingMetric(BaseMetric):
    default_prefix = "CityReferGrounding"

    def __init__(self, collect_device="cpu", prefix=None, output_dir=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.output_dir = output_dir or os.environ.get("CITYREFER_METRIC_DIR")

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            pred = sample["pred_sem_seg"]["data"].squeeze().cpu() == 1
            gt = sample["gt_sem_seg"]["data"].squeeze().cpu() == 1
            pred_box = foreground_box(pred)
            gt_box = foreground_box(gt)
            intersection = torch.logical_and(pred, gt).sum().item()
            union = torch.logical_or(pred, gt).sum().item()
            mask_iou = intersection / union if union else 0.0
            b_iou = box_iou(pred_box, gt_box)
            center_hit = False
            if pred_box is not None and gt_box is not None:
                cx = (pred_box[0] + pred_box[2]) / 2
                cy = (pred_box[1] + pred_box[3]) / 2
                center_hit = gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3]
            self.results.append({
                "img_path": sample.get("img_path", ""),
                "text": sample.get("text", ""),
                "group": sample.get("category_name", "unknown"),
                "mask_iou": mask_iou,
                "box_iou": b_iou,
                "pred_box": pred_box,
                "gt_box": gt_box,
                "center_hit": bool(center_hit),
                "nonempty": pred_box is not None,
                "pred_rle": encode_mask(pred),
            })

    @staticmethod
    def summarize(rows):
        n = len(rows)
        if not n:
            return {"count": 0}
        mask = np.asarray([x["mask_iou"] for x in rows])
        box = np.asarray([x["box_iou"] for x in rows])
        return {
            "count": n,
            "mask_gIoU": float(mask.mean() * 100),
            "mask_Pr@0.5": float((mask > 0.5).mean() * 100),
            "mean_box_iou": float(box.mean() * 100),
            "box_acc_0.25": float((box >= 0.25).mean() * 100),
            "box_acc_0.5": float((box >= 0.5).mean() * 100),
            "box_acc_0.75": float((box >= 0.75).mean() * 100),
            "center_hit_rate": float(np.mean([x["center_hit"] for x in rows]) * 100),
            "nonempty_prediction_rate": float(np.mean([x["nonempty"] for x in rows]) * 100),
        }

    def compute_metrics(self, results):
        overall = self.summarize(results)
        grouped = defaultdict(list)
        for row in results:
            grouped[row["group"]].append(row)
        group_summary = {key: self.summarize(value) for key, value in sorted(grouped.items())}
        if self.output_dir:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "per_sample_metrics.jsonl").open("w") as stream:
                for row in results:
                    stream.write(json.dumps(row) + "\n")
            (output_dir / "grounding_summary.json").write_text(json.dumps({
                "overall": overall,
                "groups": group_summary,
            }, indent=2) + "\n")
        metrics = dict(overall)
        for group, values in group_summary.items():
            metrics[f"{group}/count"] = values["count"]
            metrics[f"{group}/mean_box_iou"] = values["mean_box_iou"]
            metrics[f"{group}/box_acc_0.5"] = values["box_acc_0.5"]
        return metrics

