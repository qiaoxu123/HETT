#!/usr/bin/env python3
"""Summarize the frozen fine-tuned test and paired language control."""

import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/tenant2/dataext/rsrefseg2/cityrefer_test_unseen_localroi")
MAIN = ROOT / "runs/test_unseen_ft4096_e2_s0_r2/metrics/per_sample_metrics.jsonl"
PAIR = ROOT / "runs/test_unseen_paired_ft4096_e2_s0/metrics/per_sample_metrics.jsonl"
PAIR_MANIFEST = DATA / "paired_control/paired_alternative_manifest.jsonl"
BASELINE = Path(
    "/home/tenant2/Workspace/hett-experiments/38-rsrefseg2-cityrefer/"
    "runs/test_unseen_full_localroi/metrics/grounding_summary.json"
)


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def summarize(rows):
    box = np.asarray([row["box_iou"] for row in rows], dtype=np.float64)
    mask = np.asarray([row["mask_iou"] for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "mean_box_iou": float(box.mean() * 100),
        "box_acc_0.25": float((box >= 0.25).mean() * 100),
        "box_acc_0.5": float((box >= 0.5).mean() * 100),
        "center_hit_rate": float(np.mean([row["center_hit"] for row in rows]) * 100),
        "mask_gIoU": float(mask.mean() * 100),
        "mask_Pr@0.5": float((mask > 0.5).mean() * 100),
    }


def rle_iou(a, b):
    aa = mask_utils.decode(a).astype(bool)
    bb = mask_utils.decode(b).astype(bool)
    union = np.logical_or(aa, bb).sum()
    return float(np.logical_and(aa, bb).sum() / union) if union else 1.0


def center(box):
    if box is None:
        return None
    return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def main():
    annotations = read_jsonl(DATA / "cityrefer_test_unseen.jsonl")
    manifests = read_jsonl(DATA / "cityrefer_test_unseen_manifest.jsonl")
    records = read_jsonl(MAIN)
    assert len(annotations) == len(manifests) == len(records) == 5281
    enriched = []
    by_file = {}
    for ann, manifest, record in zip(annotations, manifests, records):
        assert Path(record["img_path"]).name == ann["file_name"] == manifest["file_name"]
        row = {**record, **manifest}
        enriched.append(row)
        by_file[manifest["file_name"]] = row

    paired_records = read_jsonl(PAIR)
    paired_manifest = read_jsonl(PAIR_MANIFEST)
    assert len(paired_records) == len(paired_manifest) == 192
    correct_subset = []
    pred_overlap = []
    center_shift = []
    unchanged = 0
    for alt, pair in zip(paired_records, paired_manifest):
        correct = by_file[pair["base_file_name"]]
        correct_subset.append(correct)
        pred_overlap.append(rle_iou(correct["pred_rle"], alt["pred_rle"]))
        unchanged += int(correct["pred_rle"]["counts"] == alt["pred_rle"]["counts"])
        c1, c2 = center(correct["pred_box"]), center(alt["pred_box"])
        if c1 is not None and c2 is not None:
            center_shift.append(float(np.linalg.norm(c1 - c2)))

    direct = [row for row in enriched if not row["processed"].get("surroundings")]
    context = [row for row in enriched if row["processed"].get("surroundings")]
    result = {
        "overall": summarize(enriched),
        "baseline": json.loads(BASELINE.read_text())["overall"],
        "counts": {
            "box_acc_0.25": sum(row["box_iou"] >= 0.25 for row in enriched),
            "box_acc_0.5": sum(row["box_iou"] >= 0.5 for row in enriched),
            "center_hit": sum(row["center_hit"] for row in enriched),
        },
        "direct": summarize(direct),
        "context": summarize(context),
        "paired_language_control": {
            "count": 192,
            "correct_instruction_subset": summarize(correct_subset),
            "alternate_visible_target": summarize(paired_records),
            "prediction_mask_overlap_after_text_switch": float(np.mean(pred_overlap) * 100),
            "median_prediction_mask_overlap_after_text_switch": float(np.median(pred_overlap) * 100),
            "identical_prediction_rate": float(unchanged / 192 * 100),
            "mean_predicted_box_center_shift_px": float(np.mean(center_shift)),
            "median_predicted_box_center_shift_px": float(np.median(center_shift)),
            "both_box_acc_0.25": float(np.mean([
                correct["box_iou"] >= 0.25 and alt["box_iou"] >= 0.25
                for correct, alt in zip(correct_subset, paired_records)
            ]) * 100),
        },
    }
    (ROOT / "FINETUNE_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
