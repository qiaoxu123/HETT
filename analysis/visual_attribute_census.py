#!/usr/bin/env python3
"""Census canonical CityNav instructions without loading navigation rollouts."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiagent.cityreferobject import get_city_refer_objects  # noqa: E402
from multiagent.visual_attributes.parser import flattened_attributes, parse_attributes  # noqa: E402
from multiagent.visual_attributes.taxonomy import VISUAL_CATEGORIES  # noqa: E402


DEV_SPLITS = ("train_seen", "val_seen", "val_unseen")


def census(data_root: Path, splits=DEV_SPLITS):
    if "test_unseen" in splits:
        raise ValueError("test_unseen is forbidden for benchmark development")
    objects = get_city_refer_objects(
        data_root / "cityrefer/objects.json", data_root / "cityrefer/processed_descriptions.json",
    )
    summary = {"scope": list(splits), "splits": {}, "all": {}}
    totals = defaultdict(Counter); total_rows = total_visual_1 = total_visual_2 = 0
    combinations = Counter(); reference_cooccurrence = defaultdict(lambda: Counter())
    attribute_count_histogram = Counter()
    for split in splits:
        rows = json.loads((data_root / "processed_citynav" / f"citynav_{split}.json").read_text())
        counts = defaultdict(Counter); visual_1 = visual_2 = 0; histogram = Counter(); combos = Counter()
        for row in rows:
            map_name = f"{row['area']}_block_{row['block']}"
            target = objects[map_name][int(row["object_ids"][0])]
            desc_id = int(row["ann_ids"][0]); text = target.descriptions[desc_id]
            parsed = parse_attributes(text); flat = flattened_attributes(text)
            visual = tuple(item for item in flat if item.split(":", 1)[0] in VISUAL_CATEGORIES)
            histogram[len(visual)] += 1
            visual_1 += bool(visual); visual_2 += len(visual) >= 2
            category_combo = tuple(sorted(category for category in parsed if category in VISUAL_CATEGORIES))
            if category_combo: combos["+".join(category_combo)] += 1
            for category, values in parsed.items():
                counts[category].update(values)
                totals[category].update(values)
            processed = target.processed_descriptions[desc_id]
            reference_count = len(processed.landmarks)
            for attribute in visual:
                reference_cooccurrence[attribute][str(reference_count)] += 1
        summary["splits"][split] = {
            "instructions": len(rows), "visual_attribute_ge1": visual_1,
            "visual_attribute_ge1_rate": visual_1 / len(rows), "visual_attribute_ge2": visual_2,
            "visual_attribute_ge2_rate": visual_2 / len(rows),
            "attribute_count_histogram": dict(sorted(histogram.items())),
            "attribute_counts": {key: dict(value.most_common()) for key, value in counts.items()},
            "category_combinations": dict(combos.most_common()),
        }
        total_rows += len(rows); total_visual_1 += visual_1; total_visual_2 += visual_2
        attribute_count_histogram.update(histogram); combinations.update(combos)
    summary["all"] = {
        "instructions": total_rows, "visual_attribute_ge1": total_visual_1,
        "visual_attribute_ge1_rate": total_visual_1 / total_rows,
        "visual_attribute_ge2": total_visual_2, "visual_attribute_ge2_rate": total_visual_2 / total_rows,
        "attribute_count_histogram": dict(sorted(attribute_count_histogram.items())),
        "attribute_counts": {key: dict(value.most_common()) for key, value in totals.items()},
        "category_combinations": dict(combinations.most_common()),
        "referenced_landmark_count_cooccurrence": {key: dict(value) for key, value in reference_cooccurrence.items()},
    }
    return summary


def markdown(stats):
    all_stats = stats["all"]
    lines = ["# Visual Attribute Census", "",
             "> Development census excludes `test_unseen` by policy. The existing canonical source census records 32,326 released instructions; this taxonomy census covers the 27,045 permitted train/validation instructions.", "",
             f"Instructions: **{all_stats['instructions']:,}**", "",
             f"At least one lexical visual attribute: **{all_stats['visual_attribute_ge1_rate']:.2%}** ({all_stats['visual_attribute_ge1']:,})", "",
             f"At least two lexical visual attributes: **{all_stats['visual_attribute_ge2_rate']:.2%}** ({all_stats['visual_attribute_ge2']:,})", "",
             "## Split coverage", "", "| split | instructions | ≥1 visual | ≥2 visual |", "|---|---:|---:|---:|"]
    for split, value in stats["splits"].items():
        lines.append(f"| {split} | {value['instructions']:,} | {value['visual_attribute_ge1_rate']:.2%} | {value['visual_attribute_ge2_rate']:.2%} |")
    for category, values in all_stats["attribute_counts"].items():
        lines += ["", f"## {category}", "", "| attribute | occurrences |", "|---|---:|"]
        lines += [f"| {name} | {count:,} |" for name, count in values.items()]
    lines += ["", "## Frequent visual-category combinations", "", "| combination | instructions |", "|---|---:|"]
    lines += [f"| {name} | {count:,} |" for name, count in list(all_stats["category_combinations"].items())[:30]]
    lines += ["", "## Referenced-landmark co-occurrence", "",
              "| attribute | occurrences with ≥1 referenced landmark | all occurrences | rate |",
              "|---|---:|---:|---:|"]
    for attribute, counts in all_stats["referenced_landmark_count_cooccurrence"].items():
        total = sum(counts.values()); with_reference = sum(value for key, value in counts.items() if int(key) > 0)
        lines.append(f"| {attribute} | {with_reference:,} | {total:,} | {with_reference / max(total, 1):.2%} |")
    lines += ["", "Counts are lexical matches, not claims of visual observability. Observability is assigned only after held-out image evaluation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    result = census(args.data_root)
    (args.output / "visual_attribute_stats.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (args.output / "VISUAL_ATTRIBUTE_CENSUS.md").write_text(markdown(result))


if __name__ == "__main__": main()
