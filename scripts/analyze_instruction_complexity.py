#!/usr/bin/env python3
"""Audit how often CityNav instructions need landmark vs auxiliary-object relations."""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train_seen", "val_seen", "val_unseen")

RELATION_RE = re.compile(
    r"\b(?:left|right|front|behind|back|rear|near|nearby|next to|adjacent|beside|"
    r"alongside|across|opposite|between|middle|around|surround(?:ed|ing)?|above|below|"
    r"under|over|inside|outside|past|before|after|towards?|closest|nearest|farthest|"
    r"north|south|east|west|corner|end|side|center|centre|edge|attached|connected|"
    r"against|facing|perpendicular|close|closed|within|shares?|shared)\b",
    re.IGNORECASE,
)

# Explicit enumeration or ordering of scene objects. Plain articles ("one building")
# are excluded unless an ordering/layout noun is also present.
ORDER_RE = re.compile(
    r"\b(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth|last|leftmost|rightmost|topmost|bottommost|"
    r"couple|pair|several|multiple|many)\b|"
    r"\b(?:top|bottom|front|back|middle|center|centre)\s+(?:row|lane|column|aisle)\b|"
    r"\b(?:row|lane|line|cluster|group|column|aisle)\s+of\b",
    re.IGNORECASE,
)

NUMBER_WORD_RE = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
    re.IGNORECASE,
)

# Only nouns that can serve as independent spatial anchors. Features attached to
# the target itself (roof, wall, windows, colour, shape) are intentionally absent.
AUX_ENTITY_RE = re.compile(
    r"\b(?:cars?|vehicles?|vans?|trucks?|buses|bus|bicycles?|bikes?|trees?|"
    r"buildings?|houses?|shops?|stores?|warehouses?|churches?|containers?|"
    r"empty\s+(?:spots?|spaces?)|parking\s+(?:spots?|spaces?)|"
    r"roads?|streets?|paths?|driveways?|sidewalks?|alleys?|rivers?|water|"
    r"lawns?|grass|fields?|gardens?|yards?|courtyards?|fences?)\b",
    re.IGNORECASE,
)

APPEARANCE_RE = re.compile(
    r"\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|gr[ae]y|"
    r"roof|rooftop|window|door|wall|brick|glass|round|square|rectangular|"
    r"triangular|l-shaped|u-shaped|tall|short|large|small|long|wide|narrow)\b",
    re.IGNORECASE,
)


def normalise(value):
    return re.sub(r"\s+", " ", value.lower().strip())


def replace_landmarks(text, names):
    output = text
    for name in sorted(set(names), key=len, reverse=True):
        if name:
            output = re.sub(re.escape(name), " <landmark> ", output, flags=re.IGNORECASE)
    return output


def relevant_surroundings(items):
    return [item for item in items if AUX_ENTITY_RE.search(item)]


def has_numbered_auxiliary(text, auxiliary_items):
    """Require number/order language and a likely auxiliary scene entity."""
    layout = r"(?:cars?|vehicles?|vans?|trucks?|trees?|buildings?|houses?|spots?|spaces?|rows?|lanes?|columns?|aisles?)"
    order = r"(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|leftmost|rightmost|topmost|bottommost|couple|pair|several|multiple|many)"
    number = r"(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    close_pattern = re.compile(
        rf"\b(?:{order}|{number})\b(?:\W+\w+){{0,4}}\W+\b{layout}\b|"
        rf"\b{layout}\b(?:\W+\w+){{0,4}}\W+\b(?:{order}|{number})\b",
        re.I,
    )
    if close_pattern.search(text):
        return True
    # Row/column phrases express ordering even if the parser omitted the cars.
    return bool(re.search(r"\b(?:row|lane|line|cluster|group|column|aisle)\s+of\b", text, re.I))


def classify(text, processed):
    landmarks = processed.get("landmarks", [])
    surroundings = processed.get("surroundings", [])
    without_landmarks = replace_landmarks(text, landmarks)
    auxiliary_items = relevant_surroundings(surroundings)
    has_relation = bool(RELATION_RE.search(without_landmarks))
    numbered = has_numbered_auxiliary(without_landmarks, auxiliary_items)
    # `surroundings` is the dataset's semantic extraction of objects around the
    # target. Appearance-only entries have already been removed by AUX_ENTITY_RE.
    auxiliary_spatial = bool(auxiliary_items)
    complex_aux = numbered or auxiliary_spatial

    # A named road/building plus a basic relation is enough for coarse Stage 1.
    # Address-like "on/off/at <landmark>" phrases are included even when no other
    # relation word appears.
    landmark_address = bool(landmarks and re.search(
        r"\b(?:at|on|off|by|from|toward|towards|outside|inside)\b.{0,24}<landmark>|"
        r"<landmark>.{0,24}\b(?:road|street|building|house|site|campus)\b",
        without_landmarks, re.I,
    ))
    landmark_coarse = bool(landmarks and (has_relation or landmark_address))
    simple_landmark = bool(landmark_coarse and not complex_aux)

    if complex_aux:
        category = "complex_auxiliary"
    elif simple_landmark:
        category = "simple_landmark"
    else:
        category = "other"
    return {
        "category": category,
        "has_landmark": bool(landmarks),
        "has_relation": has_relation,
        "landmark_coarse": landmark_coarse,
        "has_appearance": bool(APPEARANCE_RE.search(text)),
        "number_or_order": numbered,
        "auxiliary_spatial": auxiliary_spatial,
        "landmarks": landmarks,
        "surroundings": surroundings,
        "auxiliary_items": auxiliary_items,
    }


def load_rows(root):
    processed_all = json.load((root / "data/cityrefer/processed_descriptions.json").open())
    rows = []
    missing = 0
    for split in SPLITS:
        records = json.load((root / f"data/processed_citynav/citynav_{split}.json").open())
        for record in records:
            map_name = f"{record['area']}_block_{record['block']}"
            for object_id, annotation_id, text in zip(
                    record["object_ids"], record["ann_ids"], record["descriptions"]):
                try:
                    processed = processed_all[map_name][str(object_id)][annotation_id]
                except (KeyError, IndexError):
                    processed = {"target": "", "landmarks": [], "surroundings": []}
                    missing += 1
                row = {
                    "split": split,
                    "map": map_name,
                    "object_id": int(object_id),
                    "annotation_id": int(annotation_id),
                    "text": text,
                    "target": processed.get("target", ""),
                }
                row.update(classify(text, processed))
                rows.append(row)
    return rows, missing


def aggregate(rows):
    result = {}
    for split in (*SPLITS, "all"):
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        categories = Counter(row["category"] for row in subset)
        result[split] = {
            "total": len(subset),
            "categories": dict(categories),
            "category_percent": {
                key: round(100 * categories.get(key, 0) / len(subset), 2)
                for key in ("simple_landmark", "complex_auxiliary", "other")
            },
            "complex_subtypes": {
                "number_or_order": sum(row["number_or_order"] for row in subset),
                "auxiliary_spatial": sum(row["auxiliary_spatial"] for row in subset),
                "both": sum(row["number_or_order"] and row["auxiliary_spatial"] for row in subset),
            },
            "has_landmark": sum(row["has_landmark"] for row in subset),
            "landmark_coarse": sum(row["landmark_coarse"] for row in subset),
            "landmark_and_complex": sum(
                row["landmark_coarse"] and row["category"] == "complex_auxiliary" for row in subset
            ),
            "has_appearance": sum(row["has_appearance"] for row in subset),
        }
    return result


def write_report(path, stats, missing):
    overall = stats["all"]
    lines = [
        "# CityNav 指令复杂度审计",
        "",
        "统计范围：`train_seen + val_seen + val_unseen`；开发阶段未读取 `test_unseen`。",
        "",
        f"结论：按完整指令求解，约 **{overall['category_percent']['simple_landmark']:.2f}%** 是纯地标基础关系，约 "
        f"**{overall['category_percent']['complex_auxiliary']:.2f}%** 还包含辅助物体的数量、顺序或相对位置。"
        f"不过 **{100 * overall['landmark_coarse'] / overall['total']:.2f}%** 都有可用于粗定位的地标关系，因此第一阶段可以服务绝大多数样本；"
        f"其中 {100 * overall['landmark_and_complex'] / overall['total']:.2f}% 仍需要第二阶段在局部区域做细粒度目标选择。",
        "",
        "## 互斥分类结果",
        "",
        "| split | 总数 | 简单：地标+基础方位 | 复杂：辅助物体数量/位置 | 其他 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {"train_seen": "train_seen", "val_seen": "val_seen", "val_unseen": "val_unseen", "all": "合计"}
    for split in (*SPLITS, "all"):
        item = stats[split]
        count = item["categories"]
        pct = item["category_percent"]
        lines.append(
            f"| {labels[split]} | {item['total']:,} | "
            f"{count.get('simple_landmark', 0):,} ({pct['simple_landmark']:.2f}%) | "
            f"{count.get('complex_auxiliary', 0):,} ({pct['complex_auxiliary']:.2f}%) | "
            f"{count.get('other', 0):,} ({pct['other']:.2f}%) |"
        )
    subtype = overall["complex_subtypes"]
    lines += [
        "",
        "复杂类内部（可重叠）：",
        "",
        f"- 明确数量、序号或行列：{subtype['number_or_order']:,} 条。",
        f"- 其他物体参与空间关系：{subtype['auxiliary_spatial']:,} 条。",
        f"- 两者同时出现：{subtype['both']:,} 条。",
        f"- 有可用地标粗定位关系（无论后面是否复杂）：{overall['landmark_coarse']:,} 条 "
        f"({100 * overall['landmark_coarse'] / overall['total']:.2f}%)。",
        f"- 同时有地标粗定位和复杂辅助关系：{overall['landmark_and_complex']:,} 条 "
        f"({100 * overall['landmark_and_complex'] / overall['total']:.2f}%)。",
        "",
        "## 判定口径",
        "",
        "- **简单**：存在人工抽取的具名地标，并有左/右、前/后、附近、道路地址等基础关系；不依赖车辆、树、其他建筑、空车位等辅助物体。",
        "- **复杂**：辅助物体参与相对位置，或出现第几辆、第几排、两个/多个物体等数量与顺序关系。",
        "- **其他**：缺少明确地标关系，或主要是外观描述。颜色、屋顶、窗户和形状本身不算复杂辅助定位。",
        f"- `{missing}` 条缺少对应的 processed description，按可见文本保守分类。",
        "",
        "这是基于数据集人工预处理字段的可复现规则审计，不是人工逐条重标。抽样发现 `surroundings` 偶尔漏抽辅助物体，"
        f"因此 {overall['category_percent']['complex_auxiliary']:.2f}% 更适合作为复杂比例的保守近似。"
        "抽样文件保存在 `instruction_complexity_samples.json`。",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sample-per-category", type=int, default=80)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()
    rows, missing = load_rows(args.root)
    stats = aggregate(rows)
    (args.root / "instruction_complexity.json").write_text(json.dumps({
        "scope": list(SPLITS), "missing_processed": missing, "stats": stats,
    }, indent=2, ensure_ascii=False) + "\n")
    write_report(args.root / "RESULTS.md", stats, missing)

    random.seed(args.seed)
    sampled = defaultdict(list)
    for category in ("simple_landmark", "complex_auxiliary", "other"):
        candidates = [row for row in rows if row["category"] == category]
        for row in random.sample(candidates, min(args.sample_per_category, len(candidates))):
            sampled[category].append(row)
    (args.root / "instruction_complexity_samples.json").write_text(
        json.dumps(sampled, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
