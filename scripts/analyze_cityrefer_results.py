#!/usr/bin/env python3
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/tenant2/dataext/rsrefseg2/cityrefer_test_unseen_localroi")
MAIN = ROOT / "runs/test_unseen_full_localroi/metrics/per_sample_metrics.jsonl"
PAIR = ROOT / "runs/test_unseen_paired_language_control/metrics/per_sample_metrics.jsonl"
PAIR_MANIFEST = DATA / "paired_control/paired_alternative_manifest.jsonl"


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

    per_map = {key: summarize(value) for key, value in sorted(
        ((key, [row for row in enriched if row["map_name"] == key])
         for key in sorted({row["map_name"] for row in enriched})))
    }
    resolved = [row for row in enriched if not row["missing_landmarks"]]
    missing = [row for row in enriched if row["missing_landmarks"]]
    direct = [row for row in enriched if not row["processed"].get("surroundings")]
    context = [row for row in enriched if row["processed"].get("surroundings")]
    truncated = [row for row in enriched if row["text_truncated"]]
    roi_buckets = {
        "160m": [row for row in enriched if row["roi_side_m"] <= 160.0001],
        "160-240m": [row for row in enriched if 160.0001 < row["roi_side_m"] <= 240],
        ">240m": [row for row in enriched if row["roi_side_m"] > 240],
    }

    paired_records = read_jsonl(PAIR)
    paired_manifest = read_jsonl(PAIR_MANIFEST)
    assert len(paired_records) == len(paired_manifest) == 192
    pred_overlap = []
    center_shift = []
    unchanged = 0
    correct_subset = []
    for alt, pair in zip(paired_records, paired_manifest):
        correct = by_file[pair["base_file_name"]]
        correct_subset.append(correct)
        overlap = rle_iou(correct["pred_rle"], alt["pred_rle"])
        pred_overlap.append(overlap)
        unchanged += int(correct["pred_rle"]["counts"] == alt["pred_rle"]["counts"])
        c1, c2 = center(correct["pred_box"]), center(alt["pred_box"])
        if c1 is not None and c2 is not None:
            center_shift.append(float(np.linalg.norm(c1 - c2)))
    paired = {
        "count": len(paired_records),
        "correct_instruction_subset": summarize(correct_subset),
        "alternate_visible_target": summarize(paired_records),
        "prediction_mask_overlap_after_text_switch": float(np.mean(pred_overlap) * 100),
        "median_prediction_mask_overlap_after_text_switch": float(np.median(pred_overlap) * 100),
        "identical_prediction_rate": float(unchanged / len(paired_records) * 100),
        "mean_predicted_box_center_shift_px": float(np.mean(center_shift)),
        "median_predicted_box_center_shift_px": float(np.median(center_shift)),
        "both_box_acc_0.25": float(np.mean([
            correct["box_iou"] >= 0.25 and alt["box_iou"] >= 0.25
            for correct, alt in zip(correct_subset, paired_records)
        ]) * 100),
    }
    result = {
        "overall": summarize(enriched),
        "counts": {
            "box_acc_0.5": sum(row["box_iou"] >= 0.5 for row in enriched),
            "box_acc_0.25": sum(row["box_iou"] >= 0.25 for row in enriched),
            "center_hit": sum(row["center_hit"] for row in enriched),
            "truncated": len(truncated),
            "missing_landmark": len(missing),
        },
        "resolved_landmarks": summarize(resolved),
        "missing_landmarks": summarize(missing),
        "direct": summarize(direct),
        "context": summarize(context),
        "truncated": summarize(truncated) if truncated else {"count": 0},
        "roi_buckets": {key: summarize(value) for key, value in roi_buckets.items()},
        "per_map": per_map,
        "paired_language_control": paired,
    }
    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    o = result["overall"]
    p = result["paired_language_control"]
    report = f"""# RSRefSeg 2 × CityRefer test_unseen 结果

零样本 oracle-ROI 指代定位，完整 `test_unseen` 5,281 条。ROI 构造和限制见 `PROTOCOL.md`。

| 指标 | 结果 |
|---|---:|
| Mean Box IoU | {o['mean_box_iou']:.2f}% |
| Box Acc@0.25 | {o['box_acc_0.25']:.2f}% ({result['counts']['box_acc_0.25']}/5281) |
| Box Acc@0.50 | {o['box_acc_0.5']:.2f}% ({result['counts']['box_acc_0.5']}/5281) |
| 预测框中心命中 | {o['center_hit_rate']:.2f}% |
| Footprint mask gIoU | {o['mask_gIoU']:.2f}% |
| Footprint mask Pr@0.5 | {o['mask_Pr@0.5']:.2f}% |

结论：论文的 RefSegRS checkpoint 在 CityRefer 局部目标—地标区域上不能可靠找到语言所指目标；
Acc@0.50 只有 {o['box_acc_0.5']:.2f}%，不能直接作为 HETT 的目标检测器使用。

## 同图语言切换对照

192 个 ROI 均包含另一个有真实指令和 GT 的可见目标。把输入文本从原目标切换到该目标后：

- 新目标 Box Acc@0.25：{p['alternate_visible_target']['box_acc_0.25']:.2f}%；Acc@0.50：{p['alternate_visible_target']['box_acc_0.5']:.2f}%。
- 两条指令都在同图正确定位（Box IoU≥0.25）：{p['both_box_acc_0.25']:.2f}%。
- 换文本前后预测 mask 平均重叠：{p['prediction_mask_overlap_after_text_switch']:.2f}%；预测框中心平均移动 {p['mean_predicted_box_center_shift_px']:.1f}px。

这说明输出会随文本变化，但变化没有切换到语言指定的另一个可见目标；失败不是单纯“完全忽略文本”，
而是 RefSegRS 的短模板/对象分布无法零样本迁移到 CityRefer 的长关系指令和建筑场景。

## 数据接口限制

- 26/5,281（0.49%）指令超过 SigLIP2 的 64-token 上限，按 tokenizer 从尾部截断。
- 260/5,281（4.92%）至少一个地标名无法解析成同图具名对象；这部分单列，不影响主结果保留全测试集。
- GT 是 CityRefer 平面 footprint，不是逐像素遥感语义 mask，因此 Box 指标是主指标。
"""
    (ROOT / "RESULTS.md").write_text(report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

