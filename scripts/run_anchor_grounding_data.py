#!/usr/bin/env python3
"""Anchor hypotheses and relation geometry for the fixed CityRefer samples.

Reuses the entity round's manifests verbatim -- same instruction, target,
candidate ids and order, same split, episode and step -- and adds, per sample:

* the parsed instruction, split into target / anchors / relations / attributes;
* for every anchor phrase, the top-K entities of the **whole block** that could
  be it, under the lexical baseline and under frozen text-encoder similarity;
* the relation geometry and the rule score for every (candidate, anchor) pair;
* the anchor an instruction names, where exactly one non-target entity matches,
  so that anchor grounding has an evaluable subset.

The grounder never sees the target.  ``gt_anchor`` is written for evaluation and
is produced by a function that is documented as such; nothing downstream of this
file reads it except the oracle rows of the report.
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
from sensaturban_fpv.anchor_grounding import (  # noqa: E402
    TYPE_NAMES, block_nodes, between_score, gt_anchor_for, lexical_anchor_scores,
    oriented_relation_score, relation_features, relation_geometry,
    semantic_anchor_scores, anchor_log_probabilities,
)
from sensaturban_fpv.anchor_parser import (  # noqa: E402
    parse_instruction, vocabulary_stats,
)
from sensaturban_fpv.config import artifact_dir, load_config, load_landmarks  # noqa: E402
from run_gate_b import encode_texts, load_encoder  # noqa: E402

TOP_K = 10


def embed_cached(processor, model, torch, device, texts, cache):
    missing = sorted({t for t in texts if t and t not in cache})
    if missing:
        feats = encode_texts(processor, model, torch, missing, device)
        for text, feat in zip(missing, feats):
            cache[text] = feat.detach().cpu().numpy().astype(np.float32)
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base", default=None,
                    help="the entity round's sample directory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = Path(args.base) if args.base else \
        artifact_dir(cfg) / "entity_grounding" / "data"
    out_dir = Path(args.out) if args.out else \
        artifact_dir(cfg) / "language_anchor_grounding" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_by_map = load_landmarks(cfg)
    processor, model, torch = load_encoder(cfg, args.device)

    episodes = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        for ep in citynav.load_split(Path(cfg["paths"]["citynav_dir"]), split):
            episodes[(split, ep.index)] = ep

    records = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        for path in sorted(base.glob(f"{split}*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kept"):
                    records[(split, row["episode_index"], row["step"])] = row
    print(f"{len(records)} samples", flush=True)

    text_cache, node_cache = {}, {}
    type_embeddings = None
    stats = Counter()
    written = 0
    t0 = time.time()
    out_path = out_dir / "anchors.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["split"], row["episode_index"], row["step"]))
        print(f"resuming with {len(done)} done", flush=True)

    for key in sorted(records):
        if key in done:
            continue
        record = records[key]
        split, episode_index, step = key
        map_name = record["map"]
        if map_name not in node_cache:
            node_cache[map_name] = block_nodes(objects_by_map[map_name])
            embed_cached(processor, model, torch, args.device,
                         [n.name for n in node_cache[map_name] if n.name],
                         text_cache)
        nodes = node_cache[map_name]
        if type_embeddings is None:
            embed_cached(processor, model, torch, args.device, TYPE_NAMES, text_cache)
            type_embeddings = {t: text_cache[t] for t in TYPE_NAMES}

        parsed = parse_instruction(record["instruction"])
        archive = np.load(base / record["file"], allow_pickle=True)
        candidate_ids = [int(v) for v in archive["entity_ids"]]
        target_index = int(np.flatnonzero(archive["is_target"])[0])
        target_id = candidate_ids[target_index]
        episode = episodes[(split, episode_index)]
        uav_position = np.asarray(episode.trajectory[step, :3], dtype=np.float64)
        uav_yaw = float(episode.yaw()[step])

        node_xyz = np.stack([n.position for n in nodes])
        node_dim = np.stack([n.dimension for n in nodes])
        cand_rows = [i for i, n in enumerate(nodes) if n.entity_id in set(candidate_ids)]
        row_of_id = {nodes[i].entity_id: i for i in cand_rows}

        anchors = [a for a in parsed["anchors"]][:4]   # at most four hypotheses
        if not anchors:
            stats["no_anchor"] += 1
            payload = {"split": split, "map": map_name,
                       "episode_index": episode_index, "step": step,
                       "key": record["file"], "instruction": record["instruction"],
                       "target_index": target_index, "candidate_ids": candidate_ids,
                       "parsed": parsed, "n_anchors": 0,
                       "gt_anchor_row": -1, "gt_anchor_node": -1}
            with out_path.open("a") as handle:
                handle.write(json.dumps(payload) + "\n")
            continue

        embed_cached(processor, model, torch, args.device,
                     [a["phrase"] for a in anchors], text_cache)
        name_embeddings = {n.norm_name: text_cache[n.name]
                           for n in nodes if n.name and n.name in text_cache}
        name_embeddings = {n.norm_name: text_cache.get(n.name) for n in nodes
                           if n.name and text_cache.get(n.name) is not None}

        A = len(anchors)
        C = len(candidate_ids)
        anchor_ids = np.full((A, args.top_k), -1, dtype=np.int64)
        anchor_logp = np.zeros((A, args.top_k), dtype=np.float32)
        a0_ids = np.full((A, args.top_k), -1, dtype=np.int64)
        anchor_xyz = np.zeros((A, args.top_k, 3), dtype=np.float32)
        anchor_dim = np.zeros((A, args.top_k, 3), dtype=np.float32)
        # Per (candidate, anchor hypothesis, top-k) rather than per
        # (candidate, anchor): the report has to compare committing to the best
        # anchor with marginalising over several, and that comparison is only
        # meaningful if both are computed from the same table.
        rel_feat = np.zeros((C, A, args.top_k, 25), dtype=np.float32)
        rule_score = np.zeros((C, A, args.top_k), dtype=np.float32)

        for a_i, anchor in enumerate(anchors):
            lex = lexical_anchor_scores(anchor["phrase"], nodes, anchor["type"])
            sem = semantic_anchor_scores(text_cache[anchor["phrase"]],
                                         type_embeddings, name_embeddings, nodes)
            # The deployable scores are kept separate so the report can compare
            # what the lexical baseline alone achieves with what the encoder adds.
            combined = lex + 1.5 * np.clip(sem, -1.0, 1.0)
            order, logp = anchor_log_probabilities(combined, top_k=args.top_k)
            order0, _ = anchor_log_probabilities(lex, top_k=args.top_k)
            if order.size:
                anchor_ids[a_i, :len(order)] = [nodes[i].entity_id for i in order]
                anchor_logp[a_i, :len(logp)] = logp
                anchor_xyz[a_i, :len(order)] = node_xyz[order]
                anchor_dim[a_i, :len(order)] = node_dim[order]
            if order0.size:
                a0_ids[a_i, :len(order0)] = [nodes[i].entity_id for i in order0]

            relation = {"phrase": anchor["relation"], "family": anchor["family"],
                        "frame": anchor["frame"]}
            for c_i, cid in enumerate(candidate_ids):
                node = nodes[row_of_id[cid]]
                for k in range(min(args.top_k, len(order))):
                    geom = relation_geometry(node.position, anchor_xyz[a_i, k],
                                             uav_position, uav_yaw)
                    rel_feat[c_i, a_i, k] = relation_features(
                        geom, relation, node.dimension, anchor_dim[a_i, k])
                    rule_score[c_i, a_i, k] = oriented_relation_score(
                        geom, relation)
                if anchor["family"] == "between" and len(order) >= 2:
                    # "between the church and the road" needs two anchors, and
                    # the second is the next anchor phrase in the sentence.
                    if a_i + 1 < A:
                        for k in range(min(args.top_k, len(order))):
                            geom = {"between": between_score(
                                node.position, anchor_xyz[a_i, k],
                                anchor_xyz[a_i + 1, k], uav_position, uav_yaw)}
                            rule_score[c_i, a_i, k] = geom["between"]

        gt = gt_anchor_for(anchors, nodes, target_id)
        gt_row = -1
        gt_node_id = -1
        gt_xyz = np.zeros(3, dtype=np.float32)
        if gt is not None:
            node_index, anchor_row = gt
            gt_row, gt_node_id = anchor_row, nodes[node_index].entity_id
            gt_xyz = node_xyz[node_index].astype(np.float32)
            stats["gt_anchor_found"] += 1
            if gt_node_id in row_of_id:
                stats["gt_anchor_is_candidate"] += 1
            # The GT-anchor relation geometry is recomputed at evaluation time
            # from `gt_anchor_xyz`; nothing is stored per candidate for it,
            # because the oracle rows are diagnostics and not a method.
        payload = {
            "split": split, "map": map_name, "episode_index": episode_index,
            "step": step, "key": record["file"],
            "instruction": record["instruction"],
            "target_index": target_index, "candidate_ids": candidate_ids,
            "parsed": parsed, "n_anchors": A,
            "anchor_ids": anchor_ids.tolist(),
            "anchor_logp": anchor_logp.tolist(),
            "anchor_a0_ids": a0_ids.tolist(),
            "anchor_xyz": anchor_xyz.tolist(),
            "anchor_dim": anchor_dim.tolist(),
            "rel_feat": rel_feat.tolist(),
            "rule_score": rule_score.tolist(),
            "gt_anchor_row": int(gt_row), "gt_anchor_node": int(gt_node_id),
            "gt_anchor_xyz": gt_xyz.tolist(),
            "uav_position": uav_position.tolist(), "uav_yaw": uav_yaw,
            # Per-candidate geometry and the true anchor's box, so the oracle
            # rows can build the same features the deployable arms use rather
            # than a different rule function.
            # In candidate-list order, not block order: the oracle rows index
            # these by candidate position, and a block-order array would pair
            # every candidate with a different entity's geometry.
            "candidate_positions": [node_xyz[row_of_id[c]].tolist()
                                    for c in candidate_ids],
            "candidate_dimensions": [node_dim[row_of_id[c]].tolist()
                                     for c in candidate_ids],
            "gt_anchor_dim": (node_dim[[i for i, n in enumerate(nodes)
                                        if n.entity_id == gt_node_id][0]].tolist()
                              if gt_node_id >= 0 else [0.0, 0.0, 0.0]),
        }
        with out_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        written += 1
        if (written) % 100 == 0:
            print(f"  {written} ({time.time() - t0:.0f}s) gt_anchor "
                  f"{stats['gt_anchor_found']}", flush=True)

    stats["written"] = written
    (out_dir / "anchor_stats.json").write_text(json.dumps(
        {"stats": dict(stats),
         "vocabulary": vocabulary_stats([r["instruction"] for r in
                                         records.values()])},
        indent=2, default=float) + "\n")
    print(f"{dict(stats)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
