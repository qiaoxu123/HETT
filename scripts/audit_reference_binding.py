#!/usr/bin/env python3
"""Audit what CityNav actually hands over as "referenced landmarks".

This round was briefed on the premise that inference may legally consume the
data's referenced-landmark field, and that the anchor candidate set should
therefore be taken from it rather than searched for across the block.  Before
building anything on that, the premise has to be checked against the files, and
that is all this script does.  It trains nothing.

Three things are established here.

**``object_ids`` is the target, not the anchor set.**  Every one of the 27,045
records in the three splits carries ``|object_ids| == 1``, and that object's
position is always one of the episode's ``target_positions``.  The upstream code
agrees: ``MTurkTrajectory.object_id`` returns ``object_ids[0]`` and
``generate.py`` builds ``Episode(objects[map][object_id], ...)``.  So the field
named as if it held references holds the referent, and using it as an anchor
candidate set would be the target-id leak the brief forbids.

**What is legitimately available is the landmark *names*.**  ``CityReferObject``
carries ``processed_descriptions[i]`` with ``target``, ``landmarks`` and
``surroundings``, and upstream exposes them as
``Episode.description_landmarks`` / ``description_surroundings``.  That is a
first-class API, its content is a list of anchor name strings, and it does not
name the target.  That is what this round consumes.

**Reaching it is keyed on the target object.**  The annotations are stored under
the target's id, so ``description_landmarks`` is only reachable once you know
which object the sentence describes -- and that is why the audit has to say so
rather than let it pass unremarked.  The names themselves are already in the
sentence; what the annotation supplies is the *segmentation* into anchors, so the
information added is the parser's correctness and nothing else.  The round keeps
both paths apart: every arm is reported once with parser-derived anchors and once
with annotation-derived anchors.

Outputs ``artifacts/relation_v2/reference_audit.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv import citynav  # noqa: E402
from sensaturban_fpv.anchor_parser import normalise  # noqa: E402
from sensaturban_fpv.config import artifact_dir, load_config, load_landmarks  # noqa: E402

SPLITS = ("train_seen", "val_seen", "val_unseen")


def squash(text: str) -> str:
    """Whitespace-insensitive form; the two files differ only in runs of spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def load_manifest(base: Path) -> dict:
    """The entity round's kept samples, keyed as everywhere else in this project."""
    records = {}
    for split in SPLITS:
        for path in sorted(base.glob(f"{split}*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kept"):
                    records[(split, row["episode_index"], row["step"])] = row
    return records


def name_index(objects) -> tuple:
    """Normalised name -> entity ids, and the set of names, for one block."""
    by_name = defaultdict(list)
    for obj in objects.values():
        if obj.name:
            by_name[normalise(obj.name)].append(int(obj.id))
    return by_name


def bind_phrase(phrase: str, objects, by_name, type_words) -> dict:
    """Best binding of one landmark name onto the block's entities.

    Deliberately simple, and deliberately *not* learned: the audit's job is to
    measure how much of the binding problem is already solved by the strings,
    so that the round can say how much is left for a model.  A learned matcher
    here would absorb the very quantity being measured.
    """
    norm = normalise(phrase)
    out = {"phrase": phrase, "norm": norm, "exact": [], "token": [], "type": []}
    if not norm:
        return out
    for eid in by_name.get(norm, []):
        out["exact"].append(eid)
    tokens = {w for w in norm.split() if len(w) > 2}
    for obj in objects.values():
        if not obj.name:
            continue
        other = normalise(obj.name)
        if other == norm:
            continue
        shared = tokens & {w for w in other.split() if len(w) > 2}
        if shared:
            out["token"].append((int(obj.id), len(shared) / max(len(tokens), 1)))
    for obj in objects.values():
        words = type_words.get(obj.object_type, ())
        if any(w in norm for w in words):
            out["type"].append(int(obj.id))
    # One rung of the ladder: exact beats partial-token beats type-only.
    if len(out["exact"]) == 1:
        out["rung"], out["candidates"] = "exact", out["exact"]
    elif len(out["exact"]) > 1:
        out["rung"], out["candidates"] = "exact_ambiguous", out["exact"]
    elif out["token"]:
        best = max(s for _, s in out["token"])
        cands = [i for i, s in out["token"] if s >= best - 1e-9]
        out["rung"] = "token" if len(cands) == 1 else "token_ambiguous"
        out["candidates"] = cands
    elif len(out["type"]) == 1:
        out["rung"], out["candidates"] = "type_only", out["type"]
    elif out["type"]:
        out["rung"], out["candidates"] = "type_ambiguous", out["type"]
    else:
        out["rung"], out["candidates"] = "none", []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", default=None, help="the entity round's sample directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = Path(args.base) if args.base else \
        artifact_dir(cfg) / "entity_grounding" / "data"
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "relation_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_by_map = load_landmarks(cfg)
    from sensaturban_fpv.anchor_parser import TYPE_WORDS

    records = load_manifest(base)
    print(f"{len(records)} manifest samples", flush=True)

    episodes = {}
    for split in SPLITS:
        for ep in citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split):
            episodes[(split, ep.index)] = ep

    # ---- 1. does object_ids hold the target, on the records we actually use?
    raw_dir = Path(cfg["paths"]["citynav_dir"])
    raw = {}
    for split in SPLITS:
        raw[split] = json.loads((raw_dir / citynav.SPLIT_FILES[split]).read_text())

    target_match, sizes, ann_diff = 0, Counter(), 0
    for split in SPLITS:
        objs = objects_by_map
        for r in raw[split]:
            map_name = f"{r['area']}_block_{r['block']}"
            sizes[len(r["object_ids"])] += 1
            if r["object_ids"] != r["ann_ids"]:
                ann_diff += 1
            # ``get_city_refer_objects`` re-keys the block by int, not by the
            # string the JSON uses.
            oid = int(r["object_ids"][0])
            if oid in objs.get(map_name, {}):
                # ``position`` is a Point3D namedtuple, which numpy will happily
                # turn into an object array rather than three floats.
                pos = np.array(tuple(objs[map_name][oid].position), dtype=np.float64)
                tp = np.asarray(r["target_positions"], dtype=np.float64).reshape(-1, 3)
                if any(np.allclose(pos, t, atol=0.01) for t in tp):
                    target_match += 1
    total = sum(sizes.values())
    print(f"object_ids: |.| distribution {dict(sizes)}; "
          f"{target_match}/{total} are target_positions", flush=True)

    # ---- 2. per-sample annotation and binding
    rungs = Counter()
    n_landmarks = Counter()
    n_surround = Counter()
    case_counts = Counter()
    reason_counter = Counter()
    multi_same_type = 0
    multi_similar_name = 0
    order_consistent = 0
    order_checked = 0
    ann_missing = 0
    desc_mismatch = 0
    swap_examples = 0
    one_to_many = 0
    n_entities = 0
    verbatim_hits = 0
    verbatim_total = 0
    per_split_case = defaultdict(Counter)
    per_split_landmarks = defaultdict(Counter)
    rows = []

    for key in sorted(records):
        split, episode_index, step = key
        episode = episodes.get((split, episode_index))
        if episode is None or not episode.object_ids:
            ann_missing += 1
            continue
        map_name = episode.map_name
        objects = objects_by_map.get(map_name, {})
        target_id = int(episode.object_ids[0])
        desc_id = int(episode.ann_ids[0]) if episode.ann_ids else 0
        obj = objects.get(target_id)
        if obj is None or not obj.processed_descriptions:
            ann_missing += 1
            continue
        if desc_id >= len(obj.processed_descriptions):
            ann_missing += 1
            continue
        if squash(obj.descriptions[desc_id] if desc_id < len(obj.descriptions)
                  else "") != squash(episode.description):
            desc_mismatch += 1
        ann = obj.processed_descriptions[desc_id]
        landmarks = [str(x) for x in ann.landmarks]
        surroundings = [str(x) for x in ann.surroundings]
        n_landmarks[len(landmarks)] += 1
        n_surround[len(surroundings)] += 1

        by_name = name_index(objects)
        binds = [bind_phrase(p, objects, by_name, TYPE_WORDS) for p in landmarks]
        for b in binds:
            rungs[b["rung"]] += 1

        # multi-reference structure
        if len(landmarks) >= 2:
            types = []
            for b in binds:
                cands = b["candidates"]
                if cands:
                    types.append(objects[cands[0]].object_type)
            if len(types) >= 2 and len(set(types)) == 1:
                multi_same_type += 1
            toks = [set(normalise(p).split()) for p in landmarks]
            if any(toks[i] & toks[j] for i in range(len(toks))
                   for j in range(i + 1, len(toks))):
                multi_similar_name += 1

        # mention order: is the annotation's list in the order the sentence says?
        if len(landmarks) >= 2:
            text = normalise(episode.description)
            positions = []
            for p in landmarks:
                head = normalise(p)
                idx = text.find(head)
                if idx < 0:
                    # fall back to the last content word of the name
                    for w in reversed(head.split()):
                        if len(w) > 3:
                            idx = text.find(w)
                            break
                positions.append(idx)
            if all(p >= 0 for p in positions):
                order_checked += 1
                if positions == sorted(positions):
                    order_consistent += 1

        # Two ambiguities are worth separating, because only one of them is a
        # binding problem.  A name shared by many entities ("Walsall Road" is
        # every segment of one road) is one-to-many within a phrase, and a
        # reasoner can marginalise over it.  A phrase whose candidate set
        # overlaps another phrase's is the "church or library?" problem, and
        # that one needs an assignment.
        sets = [set(b["candidates"]) for b in binds]
        swap_possible = any(sets[i] & sets[j]
                            for i in range(len(sets))
                            for j in range(i + 1, len(sets)))
        if swap_possible:
            swap_examples += 1
        if not landmarks:
            case = "NO_REFERENCE"
        elif any(b["rung"] == "none" for b in binds):
            case = "E"
        elif len(landmarks) >= 2 and swap_possible:
            case = "D"
        elif any(b["rung"] in ("token", "type_only", "token_ambiguous",
                               "type_ambiguous") for b in binds):
            case = "C"
        else:
            case = "B"
        case_counts[case] += 1
        if any(b["rung"] == "exact_ambiguous" for b in binds):
            one_to_many += 1
        n_entities += sum(len(b["candidates"]) for b in binds)

        # Q3: does the annotation's own name string appear in the sentence the
        # annotator wrote it for?  A mismatch here would mean the annotation and
        # the instruction are not in the correspondence we are assuming.
        text = normalise(episode.description)
        for lm in landmarks:
            verbatim_total += 1
            if normalise(lm) in text:
                verbatim_hits += 1
        per_split_case[split][case] += 1
        per_split_landmarks[split][len(landmarks)] += 1

        rows.append({
            "split": split, "key": records[key]["file"], "map": map_name,
            "episode_index": episode_index, "step": step,
            "target_id": target_id, "desc_id": desc_id,
            "landmarks": landmarks, "surroundings": surroundings,
            "binds": [{k: v for k, v in b.items() if k != "candidates"}
                      for b in binds],
            "case": case,
        })

    # ---- 3. landmark-name vocabulary: do annotation names match entity names?
    all_names = set()
    for objects in objects_by_map.values():
        for obj in objects.values():
            if obj.name:
                all_names.add(normalise(obj.name))
    lm_hits = 0
    lm_total = 0
    for row in rows:
        for lm in row["landmarks"]:
            lm_total += 1
            if normalise(lm) in all_names:
                lm_hits += 1

    report = {
        "samples": len(rows),
        "object_ids": {
            "records": total,
            "size_distribution": {str(k): v for k, v in sorted(sizes.items())},
            "position_is_a_target_position": target_match,
            "object_ids_differs_from_ann_ids": ann_diff,
        },
        "availability": {
            "annotation_missing": ann_missing,
            "description_text_mismatch": desc_mismatch,
            "landmarks_per_instruction": {str(k): v for k, v in sorted(n_landmarks.items())},
            "surroundings_per_instruction": {str(k): v for k, v in sorted(n_surround.items())},
        },
        "binding_rungs": dict(rungs),
        "cases": dict(case_counts),
        "multi_reference": {
            "n_multi": sum(v for k, v in n_landmarks.items() if k >= 2),
            "same_type": multi_same_type,
            "shares_a_name_token": multi_similar_name,
            "order_checked": order_checked,
            "order_matches_mention": order_consistent,
            "candidate_sets_overlap": swap_examples,
        },
        "ambiguity": {
            "samples_with_a_one_to_many_name": one_to_many,
            "mean_candidate_entities_per_sample": n_entities / max(len(rows), 1),
        },
        "landmark_name_vocabulary": {
            "landmark_strings": lm_total,
            "matching_an_entity_name_exactly": lm_hits,
            "rate": lm_hits / max(lm_total, 1),
            "appearing_verbatim_in_the_instruction": verbatim_hits,
            "verbatim_rate": verbatim_hits / max(verbatim_total, 1),
        },
        "per_split_cases": {s: dict(c) for s, c in per_split_case.items()},
        "per_split_landmark_counts": {
            s: {str(k): v for k, v in sorted(c.items())}
            for s, c in per_split_landmarks.items()},
        "rows": rows,
    }
    (out_dir / "reference_audit.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")

    print(f"cases {dict(case_counts)}", flush=True)
    print(f"rungs {dict(rungs)}", flush=True)
    print(f"landmarks/instruction {dict(sorted(n_landmarks.items()))}", flush=True)
    print(f"annotation missing {ann_missing}, desc mismatch {desc_mismatch}", flush=True)
    print(f"landmark strings matching an entity name {lm_hits}/{lm_total}", flush=True)
    print(f"-> {out_dir / 'reference_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
