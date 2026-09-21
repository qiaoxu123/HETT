#!/usr/bin/env python3
"""Create same-image, alternate-visible-target language controls."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.build_cityrefer_refseg_eval import encode_rle, rasterize_target


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-map", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    repo = args.repo.resolve()
    objects = json.loads((repo / "data/cityrefer/objects.json").read_text())
    annotations = [json.loads(line) for line in (args.eval_root / "cityrefer_test_unseen.jsonl").open()]
    manifests = [json.loads(line) for line in (args.eval_root / "cityrefer_test_unseen_manifest.jsonl").open()]
    by_map_object = defaultdict(lambda: defaultdict(list))
    for ann, manifest in zip(annotations, manifests):
        by_map_object[manifest["map_name"]][manifest["object_id"]].append((ann, manifest))

    output_annotations = []
    output_manifest = []
    for map_name in sorted(by_map_object):
        source = [pair for values in by_map_object[map_name].values() for pair in values]
        step = max(1, len(source) // args.per_map)
        selected = source[::step][:args.per_map]
        for base_ann, base in selected:
            left, bottom, right, top = base["roi_world"]
            tx = np.mean([point[0] for point in base["target_contour_world"]])
            ty = np.mean([point[1] for point in base["target_contour_world"]])
            candidates = []
            for object_id, pairs in by_map_object[map_name].items():
                if object_id == base["object_id"]:
                    continue
                obj = objects[map_name][str(object_id)]
                xy = np.asarray(obj["contour"], dtype=np.float64)
                if xy[:, 0].min() < left or xy[:, 0].max() > right or xy[:, 1].min() < bottom or xy[:, 1].max() > top:
                    continue
                cx, cy = xy.mean(axis=0)
                candidates.append(((cx - tx) ** 2 + (cy - ty) ** 2, object_id, pairs[0], obj))
            if not candidates:
                continue
            _, donor_id, (donor_ann, donor_manifest), donor_object = min(candidates, key=lambda item: item[0])
            donor_mask = rasterize_target(donor_object["contour"], base["roi_world"], 512)
            output_annotations.append({
                "split": "test_unseen_paired_alt",
                "image_id": str(len(output_annotations)),
                "sent": donor_ann["sent"],
                "file_name": base_ann["file_name"],
                "segmentation": encode_rle(donor_mask),
                "category_name": "paired_visible_alternative",
            })
            output_manifest.append({
                "pair_index": len(output_manifest),
                "map_name": map_name,
                "base_file_name": base_ann["file_name"],
                "base_object_id": base["object_id"],
                "base_instruction": base_ann["sent"],
                "alternate_object_id": donor_id,
                "alternate_instruction": donor_ann["sent"],
                "alternate_source_file": donor_ann["file_name"],
                "roi_world": base["roi_world"],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ann_path = args.output_dir / "paired_alternative.jsonl"
    manifest_path = args.output_dir / "paired_alternative_manifest.jsonl"
    with ann_path.open("w") as stream:
        for item in output_annotations:
            stream.write(json.dumps(item) + "\n")
    with manifest_path.open("w") as stream:
        for item in output_manifest:
            stream.write(json.dumps(item) + "\n")
    print(json.dumps({
        "samples": len(output_annotations),
        "maps": sorted({item["map_name"] for item in output_manifest}),
        "annotation_file": str(ann_path.resolve()),
        "manifest_file": str(manifest_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

