#!/usr/bin/env python3
"""Per-sample relation geometry for the reference-binding round.

Reuses the entity round's manifests verbatim -- same instruction, target,
candidate ids and order, same split, episode and step -- and adds, per sample:

* the anchor phrases, taken from the annotation's ``landmarks`` list (the legal
  input established in ``REFERENCE_BINDING_AUDIT_V2.md``), each bound to the
  block entities carrying that name;
* the same thing again from the parser, so the two paths can be told apart;
* for every (candidate, anchor phrase, anchor entity) triple, the geometry a
  relation is read off -- in the global frame, the agent frame, and relative to
  the anchor's own footprint axis;
* the ground-truth target's row, and which anchor phrase the instruction's first
  relation attaches to, for the self-test's oracle arm only.

Two design notes that decide how the rest of the round reads.

**An anchor phrase binds to a set, not to an entity.**  A name like "Aldridge
Road" is 180 separate road segments, so the binder's output is a set of entity
ids with a confidence each, and the reasoner marginalises over it.  Nothing here
collapses that set to one entity, because doing so would hide the disambiguation
that the audit found to be the actual binding problem.

**The anchor's own frame is computed, not assumed.**  ``anchor_axis`` is the
principal axis of the anchor's footprint contour.  For a building that is the
wall direction, and it is what lets "in front of the church" be read against the
church rather than against the camera; whether that reading is the one the corpus
uses is a question for ``audit_relation_semantics.py``, which measures all three
frames on ``train_seen``/``val_seen`` and picks one per relation family.

The ground-truth target row is written for evaluation.  Nothing downstream reads
it except the oracle rows and the self-test's "given the correct anchor"
condition, and both are labelled wherever they are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv import citynav  # noqa: E402
from sensaturban_fpv.anchor_parser import normalise, parse_instruction  # noqa: E402
from sensaturban_fpv.config import artifact_dir, load_config, load_landmarks  # noqa: E402
from sensaturban_fpv.relation_v2 import (  # noqa: E402
    GEOM_DIM, geometry, principal_axis,
)

SPLITS = ("train_seen", "val_seen", "val_unseen")
MAX_ANCHOR_ENTITIES = 24      # per phrase; the tail is road segments


def squash(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text or "").strip()


def align_relations(phrases, parsed_anchors) -> list:
    """Attach a relation to each annotation anchor phrase.

    The annotation says *which entities are referenced*; the parser says *which
    words are relations*.  They are different lists and they do not always have
    the same length, so the relation has to be looked up per phrase rather than
    zipped positionally -- zipping was silently pairing an anchor with another
    phrase's relation.  A phrase the parser did not recognise as an anchor gets
    ``None``, which the scorer carries as an explicit "unstated" relation rather
    than as a zero that would masquerade as a real word.
    """
    out = []
    for phrase in phrases:
        norm = normalise(phrase)
        best, best_score = None, 0.0
        for anchor in parsed_anchors:
            other = normalise(anchor["phrase"])
            if not other:
                continue
            if other in norm or norm in other:
                score = min(len(other), len(norm)) / max(len(other), len(norm))
            else:
                left = {w for w in norm.split() if len(w) > 2}
                right = {w for w in other.split() if len(w) > 2}
                score = 0.5 * len(left & right) / max(len(left), 1)
            if score > best_score:
                best_score, best = score, anchor
        out.append(None if best is None else
                   {"phrase": best["relation"], "family": best["family"],
                    "frame": best["frame"]})
    return out


def bind(phrase: str, objects, by_name) -> list:
    """Entity ids carrying this name, longest-first by a simple coverage score.

    Exact name first; if nothing matches exactly, fall back to the entities
    sharing the most content words, so a phrase like "national probation service
    building" still reaches "National Probation Service".
    """
    norm = normalise(phrase)
    if not norm:
        return []
    exact = by_name.get(norm, [])
    if exact:
        return exact[:MAX_ANCHOR_ENTITIES]
    tokens = {w for w in norm.split() if len(w) > 2}
    if not tokens:
        return []
    scored = []
    for obj in objects.values():
        if not obj.name:
            continue
        shared = tokens & {w for w in normalise(obj.name).split() if len(w) > 2}
        if shared:
            scored.append((len(shared) / len(tokens), int(obj.id)))
    scored.sort(key=lambda x: -x[0])
    return [i for _, i in scored[:MAX_ANCHOR_ENTITIES]]


def geometry(cand_xy, anchor_xy, anchor_axis, uav_xy, yaw, cand_dim,
             anchor_dim) -> np.ndarray:
    """The 18 numbers a relation is read off, in three frames at once.

    Kept as raw geometry rather than as a per-relation scalar so that the frame
    comparison in the audit and the learned scorer see exactly the same input;
    a scorer that had its features pre-chewed by one frame's rule could not be
    compared against another frame fairly.
    """
    c = np.asarray(cand_xy, dtype=np.float64)[:2]
    a = np.asarray(anchor_xy, dtype=np.float64)[:2]
    u = np.asarray(uav_xy, dtype=np.float64)[:2]
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([np.sin(yaw), -np.cos(yaw)])
    delta = c - a
    dist = float(np.linalg.norm(delta))
    axis = np.asarray(anchor_axis, dtype=np.float64)[:2]
    perp = np.array([-axis[1], axis[0]])
    half_span = 0.5 * float(max(anchor_dim[0], anchor_dim[1]))
    return np.array([
        delta[0] / 100.0, delta[1] / 100.0,                 # global frame
        float(np.log1p(dist)), dist / 200.0,
        float(np.sin(np.arctan2(delta[1], delta[0]))),
        float(np.cos(np.arctan2(delta[1], delta[0]))),
        float(delta @ heading) / 100.0,                     # agent frame
        float(delta @ right) / 100.0,
        float(np.linalg.norm(c - u)) / 200.0,
        (float(np.linalg.norm(a - u)) - float(np.linalg.norm(c - u))) / 100.0,
        float(delta @ axis) / 100.0,                        # anchor frame
        float(delta @ perp) / 100.0,
        float(delta @ axis) / max(half_span, 1.0),          # in anchor half-widths
        float(delta @ perp) / max(half_span, 1.0),
        float(np.log1p(max(cand_dim[0] * cand_dim[1], 0.0))) / 5.0,
        float(np.log1p(max(anchor_dim[0] * anchor_dim[1], 0.0))) / 5.0,
        float(cand_dim[2]) / 20.0, float(np.log1p(half_span)) / 5.0,
    ], dtype=np.float32)


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
    episodes = {}
    for split in SPLITS:
        for ep in citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split):
            episodes[(split, ep.index)] = ep

    records = {}
    for split in SPLITS:
        for path in sorted(base.glob(f"{split}*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kept"):
                    records[(split, row["episode_index"], row["step"])] = row
    print(f"{len(records)} samples", flush=True)

    out_path = out_dir / "relation_samples.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["split"], row["episode_index"], row["step"]))
        print(f"resuming with {len(done)} done", flush=True)

    stats = Counter()
    axes_cache = {}
    written = 0
    t0 = time.time()

    for key in sorted(records):
        if key in done:
            continue
        split, episode_index, step = key
        record = records[key]
        map_name = record["map"]
        objects = objects_by_map[map_name]
        if map_name not in axes_cache:
            axes_cache[map_name] = {int(o.id): principal_axis(o.contour)
                                    for o in objects.values()}
        axes = axes_cache[map_name]
        by_name = name_index(objects)

        episode = episodes[(split, episode_index)]
        target_id = int(episode.object_ids[0])
        obj = objects[target_id]
        desc_id = int(episode.ann_ids[0])
        ann = obj.processed_descriptions[desc_id]
        anchor_names = [str(x) for x in ann.landmarks]

        archive = np.load(base / record["file"], allow_pickle=True)
        candidate_ids = [int(v) for v in archive["entity_ids"]]
        target_index = int(np.flatnonzero(archive["is_target"])[0])
        assert candidate_ids[target_index] == target_id, (
            key, candidate_ids[target_index], target_id)

        uav_xy = np.asarray(episode.trajectory[step, :2], dtype=np.float64)
        yaw = float(episode.yaw()[step])

        parsed = parse_instruction(record["instruction"])
        parsed_phrases = [a["phrase"] for a in parsed["anchors"]][:4]
        relations = [{"phrase": a["relation"], "family": a["family"],
                      "frame": a["frame"]} for a in parsed["anchors"]][:4]

        # Two paths to the same anchor set: the annotation's own landmarks, and
        # the parser's noun phrases.  Reported separately so that the parser's
        # error is not absorbed into binding or into the reasoner.
        paths = {"annotation": anchor_names[:4], "parser": parsed_phrases}
        payload = {
            "split": split, "map": map_name, "episode_index": episode_index,
            "step": step, "key": record["file"],
            "instruction": record["instruction"],
            "candidate_ids": candidate_ids, "target_index": target_index,
            "target_id": target_id,
            "uav_xy": uav_xy.tolist(), "uav_yaw": yaw,
            "target_phrase": ann.target,
        }

        for path, phrases in paths.items():
            A = len(phrases)
            bound_ids, bound_conf = [], []
            for phrase in phrases:
                ids = bind(phrase, objects, by_name)
                bound_ids.append(ids)
                bound_conf.append([1.0] * len(ids))
            K = max((len(x) for x in bound_ids), default=0)
            anchor_ids = np.full((A, K), -1, dtype=np.int64)
            anchor_xyz = np.zeros((A, K, 3), dtype=np.float32)
            anchor_dim = np.zeros((A, K, 3), dtype=np.float32)
            anchor_axis = np.zeros((A, K, 2), dtype=np.float32)
            rel_geom = np.zeros((len(candidate_ids), A, K, GEOM_DIM), dtype=np.float32)
            for a_i, ids in enumerate(bound_ids):
                for k, eid in enumerate(ids):
                    node = objects[eid]
                    anchor_ids[a_i, k] = eid
                    anchor_xyz[a_i, k] = np.asarray(tuple(node.position),
                                                    dtype=np.float32)
                    anchor_dim[a_i, k] = np.asarray(tuple(node.dimension),
                                                    dtype=np.float32)
                    anchor_axis[a_i, k] = axes[eid].astype(np.float32)
                for c_i, cid in enumerate(candidate_ids):
                    node = objects[cid]
                    for k, eid in enumerate(ids):
                        rel_geom[c_i, a_i, k] = geometry(
                            tuple(node.position), anchor_xyz[a_i, k],
                            axes[eid], uav_xy, yaw,
                            np.asarray(tuple(node.dimension), dtype=np.float64),
                            anchor_dim[a_i, k])
            payload[f"{path}_phrases"] = phrases
            payload[f"{path}_anchor_ids"] = anchor_ids.tolist()
            payload[f"{path}_anchor_xyz"] = anchor_xyz.tolist()
            payload[f"{path}_anchor_dim"] = anchor_dim.tolist()
            payload[f"{path}_rel_geom"] = rel_geom.tolist()
            if path == "annotation":
                # One relation per *annotation* phrase, matched by the phrase
                # itself.  Not the parser's list: the two are different lengths.
                payload["annotation_relations"] = align_relations(
                    phrases, parsed["anchors"])
                payload["parsed_anchor_relations"] = relations
                payload["n_anchor_entities"] = [len(x) for x in bound_ids]
                payload["target_geom"] = [
                    geometry(tuple(objects[cid].position), anchor_xyz[a_i, k],
                             axes[int(anchor_ids[a_i, k])], uav_xy, yaw,
                             np.asarray(tuple(objects[cid].dimension),
                                        dtype=np.float64),
                             anchor_dim[a_i, k]).tolist()
                    for cid in [target_id]
                    for a_i in range(A) for k in range(K)
                    if anchor_ids[a_i, k] >= 0][:64]

        stats[f"n_anchors_{len(anchor_names)}"] += 1
        stats["multi_anchor_entities"] += int(any(
            len(x) > 1 for x in payload["annotation_anchor_ids"]))
        with out_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        written += 1
        if written % 200 == 0:
            print(f"  {written} ({time.time() - t0:.0f}s)", flush=True)

    (out_dir / "relation_data_stats.json").write_text(
        json.dumps(dict(stats), indent=1) + "\n")
    print(f"{dict(stats)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
