#!/usr/bin/env python3
"""Generate scene-template coverage statistics without reading test_unseen."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects  # noqa: E402
from multiagent.scene_grounding.dataset import ALLOWED_SPLITS  # noqa: E402
from multiagent.scene_grounding.scene_template import parse_scene_template  # noqa: E402


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    objects = get_city_refer_objects(args.data_root / "cityrefer/objects.json", args.data_root / "cityrefer/processed_descriptions.json")
    report = {"splits": {}, "total": {}}
    total = Counter()
    for split in ALLOWED_SPLITS:
        rows = json.loads((args.data_root / "processed_citynav" / f"citynav_{split}.json").read_text())
        counts = Counter()
        for row in rows:
            obj = objects[f"{row['area']}_block_{row['block']}"][int(row["object_ids"][0])]
            ann = int(row["ann_ids"][0])
            if ann >= len(obj.processed_descriptions): continue
            processed = obj.processed_descriptions[ann]
            template = parse_scene_template(obj.descriptions[ann], target_phrase=processed.target,
                                            landmark_names=processed.landmarks, surroundings=processed.surroundings,
                                            target_object_type=obj.object_type)
            counts["instructions"] += 1
            counts["with_target_attributes"] += bool(template.target.get("visual_attributes"))
            counts["with_anchor"] += bool(template.anchors)
            counts["with_context"] += bool(template.context)
            counts["with_geometry"] += bool(template.geometry)
            counts["with_ordinal"] += bool(template.ordinal)
            counts["with_any_visual"] += bool(template.visual_requirements)
        report["splits"][split] = dict(counts); total.update(counts)
    report["total"] = dict(total); report["forbidden_test_unseen_read"] = False
    (args.output / "scene_template_stats.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
