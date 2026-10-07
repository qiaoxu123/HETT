#!/usr/bin/env python3
"""Phases 6-7: the language-conditioned graph, and Gate B.

Everything before this was built to make one measurement trustworthy.  The
teacher was trained on labels that *are* geometry, so it is known to understand
the relations; the parser was shown the sentence and the ontology and nothing
else, so its relation labels are not the answer in disguise; the binding maps a
name onto a road *region* rather than guessing a segment.  This script puts the
three together and asks whether the target rises in the candidate ranking.

Scores are per candidate, marginalised over the anchors a relation might refer
to, exactly as the reference audit argued they must be: 41.6% of anchor phrases
name more than one entity, so a hard choice among them throws away the
disambiguation rather than performing it.

Arms:

* ``deepseek_graph``      -- the parsed relation, the bound anchor, the teacher
* ``shuffled_relation``   -- the same anchors and geometry under a random relation
* ``no_relation``         -- the same anchors, but only distance is read
* ``td_masked``           -- the frozen visual baseline, 0.320 from the entity round
* ``graph_plus_td``       -- the two combined with a weight chosen on val_seen

Gate B is ``deepseek_graph`` against ``shuffled_relation``: if telling the model
which relation the sentence used does not beat telling it a random one, the
language has not reached the geometry and nothing downstream can be trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.config import artifact_dir, load_config, load_landmarks  # noqa: E402
from sensaturban_fpv.spatial_graph import build_block_graph  # noqa: E402
from sensaturban_fpv.spatial_graph.deepseek_parser import ONTOLOGY  # noqa: E402
from sensaturban_fpv.spatial_graph.edge_features import (  # noqa: E402
    EDGE_DIM, EdgeContext, edge_vector, footprint_distance,
)
from sensaturban_fpv.spatial_graph.node_types import NodeKind  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_geometry import RELATION_NAMES  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_teacher import (  # noqa: E402
    aggregate, build_model, mcnemar, pairwise_auc, rank_metrics, relation_vocab,
)

SPLITS = ("val_seen", "val_unseen")
SCORABLE = set(RELATION_NAMES)
ARMS = ("deepseek_graph", "shuffled_relation", "no_relation")


def load_parses(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["key"]] = row
    return out


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    s = v.std()
    return np.zeros_like(v) if s < 1e-9 else (v - v.mean()) / s


def anchor_nodes(anchor, graph):
    """The graph nodes *one* parsed anchor could denote, regions first.

    Takes a single anchor record, not the whole parse: an earlier version passed
    the record and looked for its ``anchors`` key inside one anchor, found
    nothing, and reported zero anchors bound while the parse statistics said 608.
    """
    nodes = []
    for node_id in list(anchor.get("regions", [])) + list(anchor.get("instances", [])):
        node = graph.by_id.get(node_id)
        if node is not None:
            nodes.append(node)
    return nodes


def relation_pairs(record):
    """(relation name, anchor index) for every mappable relation."""
    out = []
    anchors = record.get("anchors") or []
    for rel in (record.get("parsed") or {}).get("relations", []) or []:
        name = rel.get("relation_normalized")
        if name not in SCORABLE:
            continue
        index = rel.get("anchor")
        if isinstance(index, str):
            index = next((i for i, a in enumerate(anchors)
                          if (a.get("raw") or {}).get("entity_name") == index), None)
        if not isinstance(index, int) or not (0 <= index < len(anchors)):
            continue
        out.append((name, index))
    return out


def score_sample(torch, teacher, vocab, graph, candidates, anchors, pairs,
                 rng=None, arm="deepseek_graph"):
    """Per-candidate score, marginalised over anchor hypotheses and relations.

    A candidate's score is a ``logsumexp`` over the (relation, anchor-entity)
    slots, not a maximum: 41.6% of anchor phrases name more than one entity, and
    marginalising is what turns that from a guess into a distribution.
    """
    n = len(candidates)
    if not pairs or not anchors:
        return None
    # A "between" sentence names two anchors and the parser reports the factor
    # as a symmetric pair of between-relations, one per anchor.  Read against a
    # single anchor the between columns are all zero, which is what the first
    # version did for 73% of the covered samples -- it scored a relation whose
    # features it had thrown away.  Here each member of the pair is read with
    # the other as its second anchor.
    between_idx = sorted({i for n, i in pairs if n == "between"})
    second_for = {}
    if len(between_idx) == 2:
        second_for = {between_idx[0]: between_idx[1],
                      between_idx[1]: between_idx[0]}

    columns = []
    for name, index in pairs:
        if arm == "shuffled_relation":
            name = str(rng.choice([r for r in RELATION_NAMES if r != name]))
        nodes = list(anchors.get(index, []))
        if not nodes:
            continue
        partner = None
        if name == "between" and index in second_for:
            others = anchors.get(second_for[index], [])
            if others:
                partner = others[0]
        rel_id = vocab.get(name, 0)
        ctx = EdgeContext(road_regions=graph.regions,
                          road_members=graph.context.road_members,
                          second_anchor=(None if partner is None
                                         else np.asarray(partner.center[:2])))
        geom = np.stack([edge_vector(node, c, ctx)
                         for node in nodes for c in candidates])
        with torch.no_grad():
            out = teacher(torch.as_tensor(geom, dtype=torch.float32),
                          torch.full((len(geom),), rel_id,
                                     dtype=torch.long)).numpy()
        out = out.reshape(len(nodes), n)
        peak = out.max(axis=0, keepdims=True)
        columns.append((peak + np.log(np.exp(out - peak).sum(axis=0,
                                                             keepdims=True))
                        ).squeeze(0))
    if not columns:
        return None
    stacked = np.stack(columns)
    peak = stacked.max(axis=0)
    return peak + np.log(np.exp(stacked - peak).sum(axis=0))


def distance_only(graph, candidates, anchors, pairs):
    """G0: distance to the anchor, no relation and no learning."""
    n = len(candidates)
    if not pairs or not anchors:
        return None
    best = np.full(n, -np.inf)
    for _, index in pairs:
        for node in anchors[index]:
            for i, c in enumerate(candidates):
                d = footprint_distance(node, c)
                best[i] = max(best[i], -d)
    return best if np.isfinite(best).any() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "spatial_graph"
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(4)

    vocab = relation_vocab(RELATION_NAMES)
    teacher = build_model(torch, len(vocab))
    state = out_dir / "relation_teacher.pt"
    if not state.exists():
        raise SystemExit("run scripts/train_relation_teacher.py first; it "
                         "saves the fitted teacher to relation_teacher.pt")
    teacher.load_state_dict(torch.load(state, map_location="cpu"))
    teacher.eval()

    objects_by_map = load_landmarks(cfg)
    rows = [json.loads(line) for line in
            (artifact_dir(cfg) / "relation_v2" / "relation_samples.jsonl")
            .read_text().splitlines() if line.strip()]
    parses = load_parses(out_dir / "deepseek_parse_val_unseen.jsonl")
    graphs, visual = {}, {}
    records = {s: [] for s in SPLITS}

    for row in rows:
        if row["split"] not in SPLITS or row["key"] not in parses:
            continue
        record = parses[row["key"]]
        if row["map"] not in graphs:
            graphs[row["map"]] = build_block_graph(objects_by_map[row["map"]],
                                                   row["map"])
        graph = graphs[row["map"]]
        candidates = [graph.by_id[c] for c in row["candidate_ids"]
                      if c in graph.by_id]
        if len(candidates) != len(row["candidate_ids"]):
            continue
        anchors = {i: anchor_nodes(a, graph)
                   for i, a in enumerate(record.get("anchors") or [])}
        pairs = relation_pairs(record)
        records[row["split"]].append({
            "key": row["key"], "map": row["map"],
            "candidates": candidates, "anchors": anchors, "pairs": pairs,
            "target_index": row["target_index"],
            "graph": graph,
        })

    print(f"samples: " + ", ".join(
        f"{s} {len(records[s])}" for s in SPLITS), flush=True)
    coverage = {"with_a_mapped_relation": 0, "with_a_bound_anchor": 0}
    for s in SPLITS:
        for r in records[s]:
            coverage["with_a_mapped_relation"] += int(bool(r["pairs"]))
            coverage["with_a_bound_anchor"] += int(
                any(r["anchors"].get(i) for _, i in r["pairs"]))
    print(f"coverage: {coverage}", flush=True)

    def run(split, arm, rng=None):
        out = []
        for r in records[split]:
            if arm == "no_relation":
                scores = distance_only(r["graph"], r["candidates"],
                                       r["anchors"], r["pairs"])
            else:
                scores = score_sample(torch, teacher, vocab, r["graph"],
                                      r["candidates"], r["anchors"], r["pairs"],
                                      rng=rng, arm=arm)
                if scores is not None and not np.isfinite(scores).any():
                    scores = None
            if scores is None:
                continue
            row = rank_metrics(np.asarray(scores, dtype=np.float64),
                               r["target_index"])
            row["key"] = r["key"]
            out.append(row)
        return out

    # Every arm is scored on the same samples: the intersection of those where
    # all of them are defined.  An arm that silently drops the samples it cannot
    # handle would otherwise be compared against a different question -- the
    # language arm covers fewer samples than the distance prior, and the two
    # numbers would not be about the same thing.
    raw_rows = {arm: {s: run(s, arm, np.random.default_rng(0)) for s in SPLITS}
                for arm in ARMS}
    common = {s: set.intersection(*[{r["key"] for r in raw_rows[a][s]}
                                    for a in ARMS]) for s in SPLITS}
    per_arm_rows = {arm: {s: [r for r in raw_rows[arm][s]
                              if r["key"] in common[s]] for s in SPLITS}
                    for arm in ARMS}
    coverage["scored_by_every_arm"] = {s: len(common[s]) for s in SPLITS}
    results = {arm: {s: aggregate(per_arm_rows[arm][s]) for s in SPLITS}
               for arm in ARMS}

    # val_seen is parsed only if a parse file exists for it; the gate is decided
    # on val_unseen, which section 8 reserves for final testing.
    print(f"\n{'arm':22s} {'val_seen':>9s} {'unseen':>9s} {'top4':>7s} {'mrr':>7s}")
    for arm in results:
        r = results[arm]["val_unseen"]
        vs = results[arm]["val_seen"]
        shown = f"{vs['top1']:9.4f}" if vs.get("n") else f"{'-':>9s}"
        if not r.get("n"):
            print(f"{arm:22s} {shown} {'(no samples)':>9s}", flush=True)
            continue
        print(f"{arm:22s} {shown} {r['top1']:9.4f} {r.get('top4', 0):7.4f} "
              f"{r.get('mrr', 0):7.4f}", flush=True)

    gate = {}
    if per_arm_rows["deepseek_graph"]["val_unseen"] and \
            per_arm_rows["shuffled_relation"]["val_unseen"]:
        a_rows = per_arm_rows["deepseek_graph"]["val_unseen"]
        b_map = {r["key"]: r["top1"] for r in
                 per_arm_rows["shuffled_relation"]["val_unseen"]}
        a = np.array([r["top1"] for r in a_rows], dtype=bool)
        b = np.array([bool(b_map.get(r["key"], 0)) for r in a_rows], dtype=bool)
        gate["deepseek_vs_shuffled"] = mcnemar(a, b)
        gate["deepseek_top1"] = results["deepseek_graph"]["val_unseen"]["top1"]
        gate["shuffled_top1"] = results["shuffled_relation"]["val_unseen"]["top1"]
        gate["conditions"] = {
            "B1_deepseek_beats_shuffled_relation": (
                gate["deepseek_top1"] > gate["shuffled_top1"]
                and gate["deepseek_vs_shuffled"]["p_value"] < 0.05),
        }
        gate["passed"] = all(gate["conditions"].values())

    report = {"coverage": coverage, "results": results, "gate": gate,
              "n": {s: len(records[s]) for s in SPLITS}}
    (out_dir / "graph_grounding_gate.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps(gate.get("conditions", {}), indent=1), flush=True)
    print(f"GATE B {'PASS' if gate.get('passed') else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
