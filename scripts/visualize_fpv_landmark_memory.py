#!/usr/bin/env python3
"""Create inspectable paired RGB/geometry panels from frozen-eval records."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def metric_top1(record):
    return int(record.get("metrics", {}).get("top1", 0) > .5)


def make_panel(row, eval_map, source, output):
    sample = row["sample_id"]
    top = eval_map.get(("topdown", 0, "")); fpv = eval_map.get(("fpv", 0, ""))
    mem = eval_map.get(("landmark_memory_top5", 0, ""))
    views = row["view_observations"]
    ids = list(map(int, row["referenced_landmark_ids"]))
    positive = ids[0] if ids else -1
    candidates = {int(c["landmark_id"]): c for c in source.get("candidates", [])}
    pred_id = max(fpv.get("candidate_scores", {}), key=lambda x: float(fpv["candidate_scores"][x])) if fpv else "?"
    mem_pred = max(mem.get("candidate_scores", {}), key=lambda x: float(mem["candidate_scores"][x])) if mem else "?"
    panels = [("Top-down", row["topdown_image"], source.get("label_file"))]
    for view_name in ("FPV", "Oblique 30°", "Oblique 45°"):
        key = {"FPV": "fpv", "Oblique 30°": "oblique30", "Oblique 45°": "oblique45"}[view_name]
        observation = views.get(key)
        if observation:
            panels.append((view_name, observation["image"], observation["label_file"]))
    for i, observation in enumerate(row.get("history_fpv", [])[1:4], start=1):
        panels.append((f"Past FPV offset {observation.get('offset', '?')}", observation["image"], observation["label_file"]))
    rendered = []
    for view_name, path, label_file in panels:
        image = cv2.imread(path)
        if image is None:
            image = np.zeros((224, 224, 3), np.uint8)
        h, w = image.shape[:2]
        if label_file:
            data = np.load(label_file)
            mask = data.get(f"candidate_{positive}")
        else:
            mask = None
        if mask is not None and np.any(mask):
            overlay = image.copy(); overlay[np.asarray(mask).astype(bool)] = (0, 0, 255)
            image = cv2.addWeighted(image, .72, overlay, .28, 0)
        rgb = cv2.cvtColor(cv2.resize(image, (320, 320)), cv2.COLOR_BGR2RGB)
        rendered.append((view_name, Image.fromarray(rgb)))
    canvas = Image.new("RGB", (990, 1180), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, im) in enumerate(rendered):
        x = 10 + (i % 3) * 330; y = 90 + (i // 3) * 340
        canvas.paste(im, (x, y)); draw.text((x, y - 20), label, fill="black")
    ref_name = candidates.get(positive, {}).get("name", str(positive))
    pred_name = candidates.get(int(pred_id), {}).get("name", str(pred_id)) if str(pred_id).lstrip("-").isdigit() else str(pred_id)
    mem_name = candidates.get(int(mem_pred), {}).get("name", str(mem_pred)) if str(mem_pred).lstrip("-").isdigit() else str(mem_pred)
    header = f"{sample} | {row['split']} | ref={ref_name} | FPV pred={pred_name} | memory pred={mem_name}"
    draw.text((10, 8), header[:110], fill="black")
    draw.text((10, 35), row["instruction"][:110], fill="black")
    draw.text((10, 1150), f"FPV Top1={metric_top1(fpv) if fpv else '?'}; memory Top1={metric_top1(mem) if mem else '?'}; red=projected reference geometry", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True); canvas.save(output, quality=90)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", type=Path, required=True); p.add_argument("--source-dataset", type=Path, required=True); p.add_argument("--evaluation", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--per-group", type=int, default=20); a=p.parse_args()
    rows = [json.loads(x) for x in (a.dataset / "manifest.jsonl").read_text().splitlines()]
    src = {json.loads(x)["sample_id"]: json.loads(x) for x in (a.source_dataset / "manifest.jsonl").read_text().splitlines()}
    metrics = [json.loads(x) for x in (a.evaluation / "per_sample.jsonl").read_text().splitlines()]
    by_sample = {}
    for r in metrics: by_sample.setdefault(r["sample_id"], {})[(r["view"], r.get("history_n", 0), r.get("aggregation", ""))] = r
    groups = {"topdown_fail_fpv_success": [], "current_fail_memory_success": [], "fpv_failure": [], "same_class_ambiguity": []}
    for row in rows:
        if row["split"] not in ("val_seen", "val_unseen"): continue
        rec = by_sample.get(row["sample_id"], {})
        top = rec.get(("topdown", 0, "")); fpv = rec.get(("fpv", 0, "")); mem = rec.get(("landmark_memory_top5", 0, ""))
        if top and fpv and not metric_top1(top) and metric_top1(fpv): groups["topdown_fail_fpv_success"].append(row)
        if fpv and mem and not metric_top1(fpv) and metric_top1(mem): groups["current_fail_memory_success"].append(row)
        if fpv and not metric_top1(fpv): groups["fpv_failure"].append(row)
        pos = next((x for x in src[row["sample_id"]].get("positives", []) if int(x["landmark_id"]) in set(map(int, row["referenced_landmark_ids"]))), {})
        same = sum(1 for c in src[row["sample_id"]].get("candidates", []) if c.get("category") == pos.get("category") and int(c["landmark_id"]) not in set(map(int, row["referenced_landmark_ids"])))
        if fpv and not metric_top1(fpv) and same > 0: groups["same_class_ambiguity"].append(row)
    counts = {}
    for name, selected in groups.items():
        counts[name] = min(len(selected), a.per_group)
        for row in selected[:a.per_group]:
            make_panel(row, by_sample.get(row["sample_id"], {}), src[row["sample_id"]], a.output / name / f"{row['split']}_{row['sample_id']}.jpg")
    (a.output / "selection_counts.json").write_text(json.dumps({k: {"available": len(groups[k]), "saved": v} for k, v in counts.items()}, indent=2) + "\n")
    print(json.dumps({k: {"available": len(groups[k]), "saved": v} for k, v in counts.items()}, indent=2))


if __name__ == "__main__": main()
