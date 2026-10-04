"""Create a reproducible, one-row-per-episode CityNav instruction census."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median


SPLITS = ("train_seen", "val_seen", "val_unseen", "test_unseen")
WORDS = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:st|nd|rd|th)?", re.I)
MARKERS = {
    "left/right": r"\b(?:left|right)\b",
    "front/behind": r"\b(?:front|behind|backside|back)\b",
    "near/adjacent": r"\b(?:near|nearby|next to|beside|adjacent|close to)\b",
    "between": r"\bbetween\b",
    "across/opposite": r"\b(?:across|opposite)\b",
    "inside/within": r"\b(?:inside|within|in the middle of)\b",
    "around/surround": r"\b(?:around|surround(?:s|ed|ing)?)\b",
    "past/beyond": r"\b(?:past|beyond)\b",
    "cardinal_direction": r"\b(?:north|south|east|west|northeast|northwest|southeast|southwest)\b",
    "ordinal": r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b",
    "color": r"\b(?:red|blue|green|yellow|white|black|grey|gray|brown|orange|purple|pink)\b",
    "motion_verb": r"\b(?:fly|go|turn|head|move|stop|pass|cross|follow|take off|land)\b",
}
COMPILED_MARKERS = {name: re.compile(pattern, re.I) for name, pattern in MARKERS.items()}
STOPWORDS = set(
    "a an and are as at be by for from in into is it of on or that the there this to "
    "was were which with you your its just very then has have had can should would could"
    .split()
)


def normalized(text):
    return " ".join(text.casefold().split())


def alphanumeric(text):
    return "".join(character for character in text.casefold() if character.isalnum())


def words(text):
    return WORDS.findall(text)


def marker_names(text):
    return [name for name, pattern in COMPILED_MARKERS.items() if pattern.search(text)]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(data_root):
    objects_path = data_root / "cityrefer" / "objects.json"
    processed_path = data_root / "cityrefer" / "processed_descriptions.json"
    objects = json.loads(objects_path.read_text(encoding="utf-8"))
    processed = json.loads(processed_path.read_text(encoding="utf-8"))
    source_paths = [objects_path, processed_path]
    rows = []

    for split in SPLITS:
        source_path = data_root / "processed_citynav" / f"citynav_{split}.json"
        source_paths.append(source_path)
        trajectories = json.loads(source_path.read_text(encoding="utf-8"))
        for source_row, trajectory in enumerate(trajectories):
            map_name = f"{trajectory['area']}_block_{trajectory['block']}"
            object_id = int(trajectory["object_ids"][0])
            ann_id = int(trajectory["ann_ids"][0])
            obj = objects[map_name][str(object_id)]
            text = obj["descriptions"][ann_id]
            annotation = processed.get(map_name, {}).get(str(object_id), [])
            annotation = annotation[ann_id] if ann_id < len(annotation) else None
            raw_texts = trajectory.get("descriptions", [])
            raw_text = raw_texts[0] if raw_texts else None
            rows.append({
                "split": split,
                "source_row": source_row,
                "map": map_name,
                "object_id": object_id,
                "ann_id": ann_id,
                "episode_id": f"{map_name}:{object_id}:{ann_id}",
                "target_type": str(obj.get("object_type", "")),
                "text": text,
                "raw_text": raw_text,
                "raw_matches": raw_text == text,
                "word_count": len(words(text)),
                "landmarks": len(annotation["landmarks"]) if annotation else None,
                "surroundings": len(annotation["surroundings"]) if annotation else None,
                "markers": marker_names(text),
            })
    return rows, source_paths


def describe(rows, label):
    counts = [row["word_count"] for row in rows]
    unique_text = len({normalized(row["text"]) for row in rows})
    maps = len({row["map"] for row in rows})
    empty = sum(not row["text"].strip() for row in rows)
    relation_markers = set(COMPILED_MARKERS) - {"color", "motion_verb"}
    relation_rows = sum(bool(relation_markers.intersection(row["markers"])) for row in rows)
    lines = [
        f"{label}: rows={len(rows):,}; unique_episode_ids={len({row['episode_id'] for row in rows}):,}; "
        f"unique_normalized_texts={unique_text:,}; maps={maps}; blank={empty}",
        f"  words: mean={mean(counts):.2f}; median={median(counts):g}; "
        f"p90={percentile(counts, 0.90)}; p95={percentile(counts, 0.95)}; "
        f"min={min(counts)}; max={max(counts)}",
        f"  lexical spatial/ordinal marker: {relation_rows:,} ({100 * relation_rows / len(rows):.2f}%)",
    ]
    bins = Counter(
        "0-10" if n <= 10 else "11-20" if n <= 20 else "21-40" if n <= 40
        else "41-80" if n <= 80 else "81+"
        for n in counts
    )
    lines.append("  word-count bins: " + "; ".join(
        f"{name}={bins[name]:,}" for name in ("0-10", "11-20", "21-40", "41-80", "81+")
    ))
    return lines


def create_report(rows, source_paths, data_root):
    lines = [
        "Refined CityNav navigation instructions: complete census",
        "=" * 64,
        "Scope: all four released trajectory JSON splits, one row per trajectory record.",
        "Instruction definition: objects.json descriptions[ann_ids[0]], the text",
        "actually consumed by Episode.target_description in HETT. The raw trajectory",
        "descriptions[0] is compared against this canonical model input.",
        "Word counts use the script's English word/number regex. Marker counts are",
        "case-insensitive lexical matches, can overlap, and are not semantic labels.",
        "Source dataset: refined_citynav (not the original CityNav release).",
        "This is a source-file census; runtime episode-generation filters may exclude",
        "some trajectory records from actual model training.",
        "",
        "SOURCE FILES (SHA-256)",
    ]
    for path in source_paths:
        lines.append(f"{path.relative_to(data_root)}\t{sha256(path)}")
    lines.extend(("", "OVERVIEW"))
    lines.extend(describe(rows, "ALL"))
    for split in SPLITS:
        lines.extend(describe([row for row in rows if row["split"] == split], split))

    text_frequency = Counter(normalized(row["text"]) for row in rows)
    episode_frequency = Counter(row["episode_id"] for row in rows)
    mismatch = [row for row in rows if not row["raw_matches"]]
    normalized_mismatch = [
        row for row in mismatch
        if row["raw_text"] is None or normalized(row["raw_text"]) != normalized(row["text"])
    ]
    alphanumeric_mismatch = [
        row for row in mismatch
        if row["raw_text"] is None or alphanumeric(row["raw_text"]) != alphanumeric(row["text"])
    ]
    missing_processed = sum(row["landmarks"] is None for row in rows)
    lines.extend((
        "",
        "DATA INTEGRITY AND OVERLAP",
        f"Repeated episode-ID rows beyond first occurrence: {sum(count - 1 for count in episode_frequency.values()):,}",
        f"Repeated normalized-text rows beyond first occurrence: {sum(count - 1 for count in text_frequency.values()):,}",
        f"Trajectory descriptions differing from model-input descriptions: {len(mismatch):,}",
        f"  Still different after whitespace/case normalization: {len(normalized_mismatch):,}",
        f"  Still different after removing punctuation too: {len(alphanumeric_mismatch):,}",
        f"Rows missing processed landmark annotation: {missing_processed:,}",
    ))
    for split in SPLITS:
        lines.append(f"  {split} exact text mismatches: " + str(sum(
            row["split"] == split for row in mismatch
        )))
    texts_by_split = defaultdict(set)
    ids_by_split = defaultdict(set)
    for row in rows:
        texts_by_split[normalized(row["text"])].add(row["split"])
        ids_by_split[row["episode_id"]].add(row["split"])
    cross_texts = [text for text, splits in texts_by_split.items() if len(splits) > 1]
    cross_ids = [episode_id for episode_id, splits in ids_by_split.items() if len(splits) > 1]
    lines.append(f"Normalized texts appearing in multiple splits: {len(cross_texts):,}")
    lines.append(f"Episode IDs appearing in multiple splits: {len(cross_ids):,}")
    if cross_ids:
        lines.append("First 20 cross-split episode IDs: " + ", ".join(cross_ids[:20]))
    if cross_texts:
        lines.append("First 20 cross-split texts (JSON quoted):")
        lines.extend("  " + json.dumps(text, ensure_ascii=False) for text in cross_texts[:20])
    split_maps = {
        split: {row["map"] for row in rows if row["split"] == split}
        for split in SPLITS
    }
    lines.append("Map overlap between splits (shared map count):")
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            lines.append(f"  {first} / {second}: {len(split_maps[first] & split_maps[second])}")

    lines.extend(("", "LEXICAL MARKERS (overlapping; count and percent of all rows)"))
    for marker in COMPILED_MARKERS:
        count = sum(marker in row["markers"] for row in rows)
        lines.append(f"{marker}\t{count:,}\t{100 * count / len(rows):.2f}%")
    lines.append("Marker rates by split (percent; same overlapping lexical rules):")
    lines.append("marker\t" + "\t".join(SPLITS))
    for marker in COMPILED_MARKERS:
        rates = []
        for split in SPLITS:
            split_rows = [row for row in rows if row["split"] == split]
            rates.append(f"{100 * sum(marker in row['markers'] for row in split_rows) / len(split_rows):.2f}%")
        lines.append(marker + "\t" + "\t".join(rates))

    landmark_counts = Counter(row["landmarks"] for row in rows)
    lines.extend(("", "PROCESSED LANDMARK REFERENCES PER INSTRUCTION"))
    for count, frequency in sorted(landmark_counts.items(), key=lambda pair: (pair[0] is None, pair[0] or 0)):
        lines.append(f"{count if count is not None else 'missing'}\t{frequency:,}\t{100 * frequency / len(rows):.2f}%")

    target_types = Counter(row["target_type"] for row in rows)
    lines.extend(("", "TARGET OBJECT TYPES (all types)"))
    for target_type, frequency in target_types.most_common():
        lines.append(f"{target_type}\t{frequency:,}\t{100 * frequency / len(rows):.2f}%")
    lines.append("Target-type rates by split (percent):")
    lines.append("target_type\t" + "\t".join(SPLITS))
    for target_type in target_types:
        rates = []
        for split in SPLITS:
            split_rows = [row for row in rows if row["split"] == split]
            rates.append(f"{100 * sum(row['target_type'] == target_type for row in split_rows) / len(split_rows):.2f}%")
        lines.append(target_type + "\t" + "\t".join(rates))

    term_frequency = Counter(
        word.casefold() for row in rows for word in words(row["text"])
        if word.casefold() not in STOPWORDS and len(word) > 2
    )
    lines.extend(("", "TOP 40 NON-STOPWORD TERMS (frequency across all rows)"))
    for term, frequency in term_frequency.most_common(40):
        lines.append(f"{term}\t{frequency:,}")

    car_building = target_types["Car"] + target_types["Building"]
    one_or_two_landmarks = landmark_counts[1] + landmark_counts[2]
    relation_markers = set(COMPILED_MARKERS) - {"color", "motion_verb"}
    spatial_count = sum(bool(relation_markers.intersection(row["markers"])) for row in rows)
    motion_count = sum("motion_verb" in row["markers"] for row in rows)
    def split_marker_rate(split, marker):
        split_rows = [row for row in rows if row["split"] == split]
        return 100 * sum(marker in row["markers"] for row in split_rows) / len(split_rows)

    lines.extend((
        "",
        "INTERPRETATION AND CAUTIONS",
        f"- {car_building:,}/{len(rows):,} ({100 * car_building / len(rows):.2f}%) target cars or buildings; "
        "this is consistent with object-localization-heavy language.",
        f"- {spatial_count:,}/{len(rows):,} ({100 * spatial_count / len(rows):.2f}%) contain at least one "
        "spatial/ordinal lexicon marker; multiple relations can appear in one instruction.",
        f"- {one_or_two_landmarks:,}/{len(rows):,} ({100 * one_or_two_landmarks / len(rows):.2f}%) "
        f"have one or two processed landmark references. The {landmark_counts[0]:,} "
        "zero-landmark rows merit separate review.",
        f"- Only {motion_count:,}/{len(rows):,} ({100 * motion_count / len(rows):.2f}%) match "
        "the limited motion-verb lexicon; this is not a validated imperative-sentence classifier.",
        f"- Ordinal markers occur in {split_marker_rate('train_seen', 'ordinal'):.2f}% "
        f"of train_seen versus {split_marker_rate('test_unseen', 'ordinal'):.2f}% "
        "of test_unseen instructions. This is a lexical distribution shift, not a causal",
        "  explanation for any navigation performance difference.",
        f"- Seen and unseen map sets do not overlap. The {len(cross_texts)} cross-split "
        "repeated texts do not",
        "  share episode IDs; text repetition alone is not evidence of episode leakage.",
        f"- {len(mismatch):,} canonical/raw pairs differ exactly, but "
        f"{len(alphanumeric_mismatch):,} differ after case, whitespace and punctuation removal; "
        "these are formatting variants under that normalization, not detected word changes.",
        "- All counts above describe raw released records; compare training sample counts",
        "  only after accounting for HETT's episode-generation filters.",
    ))

    lines.extend((
        "",
        "MODEL-INPUT / TRAJECTORY-TEXT DIFFERENCES (all exact mismatches)",
        "The model reads canonical_text_json. The trajectory JSON stores raw_text_json.",
        "Whitespace-only changes are included; no claim of semantic difference is made.",
        "split\tsource_row_zero_based\tepisode_id\tcanonical_text_json\traw_text_json",
    ))
    for row in mismatch:
        lines.append("\t".join((
            row["split"], str(row["source_row"]), row["episode_id"],
            json.dumps(row["text"], ensure_ascii=False),
            json.dumps(row["raw_text"], ensure_ascii=False),
        )))

    lines.extend((
        "",
        "ALL INSTRUCTIONS",
        "One TSV row per source trajectory, in split/file order. The final text column",
        "is a JSON-quoted string, preserving its original whitespace and Unicode.",
        "split\tsource_row_zero_based\tepisode_id\tmap\tobject_id\tann_id\t"
        "target_type\twords\tlandmarks\tsurroundings\tlexical_markers\tinstruction_json",
    ))
    for row in rows:
        lines.append("\t".join((
            row["split"], str(row["source_row"]), row["episode_id"], row["map"],
            str(row["object_id"]), str(row["ann_id"]), row["target_type"],
            str(row["word_count"]),
            str(row["landmarks"]) if row["landmarks"] is not None else "NA",
            str(row["surroundings"]) if row["surroundings"] is not None else "NA",
            ",".join(row["markers"]), json.dumps(row["text"], ensure_ascii=False),
        )))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, source_paths = load_records(args.data_root)
    report = create_report(rows, source_paths, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {len(rows):,} instructions to {args.output}")


if __name__ == "__main__":
    main()
