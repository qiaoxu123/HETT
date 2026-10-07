#!/usr/bin/env python3
"""Gate E: what the graph is worth where the language is inside the ontology.

The previous round measured the graph at 0.1008 against a distance prior's
0.1176 and could not say why.  Two explanations were available and they call for
opposite decisions: either the corpus is mostly outside the ontology, so nothing
downstream could have shown anything, or the mapping is formal and carries no
target-discriminative information even where it succeeds.  This script separates
them by restating the question on the subset where the first explanation cannot
apply.

Nothing is retrained.  The teacher is the one Gate A judged
(``relation_teacher.pt``), the parses are the cached ones, and the only new code
is classification and evaluation -- so a difference here is a difference in the
data, not in a model.

The subsets, all from §2 and §4 of the brief:

* **ONTOLOGY_COVERED** -- every spatial relation in the sentence maps.  A colour
  or a size word is not a spatial relation and does not disqualify a sample; an
  ``ASSOCIATED_WITH`` anchor likewise does not, since it asserts no geometry.
* **PARTIAL_COVERED** -- some map, at least one does not.
* **UNKNOWN_ONLY** -- none map.

and, inside ONTOLOGY_COVERED, by whether the anchors can be resolved:

* **BINDING_CLEAN** -- every anchor a mapped relation uses resolves to exactly
  one node.  One RoadRegion counts as one node, since the region *is* the entity
  the name denotes.
* **BINDING_AMBIGUOUS** -- at least one resolves to several.
* **BINDING_UNRESOLVED** -- at least one resolves to none.

This is evaluation only.  Nothing is selected on ``val_unseen``: the subsets are
defined by the parse and the map, both of which are fixed before the answer is
consulted, and no threshold or weight is fitted here.
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
    EdgeContext, edge_vector, footprint_distance,
)
from sensaturban_fpv.spatial_graph.node_types import NodeKind  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_geometry import RELATION_NAMES  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_teacher import (  # noqa: E402
    aggregate, build_model, mcnemar, pairwise_auc, rank_metrics, relation_vocab,
)

NON_SPATIAL = {"UNKNOWN", "ASSOCIATED_WITH"}
# Relations a distance prior already reproduces.  The non-distance subset exists
# to ask whether anything survives their removal.
DISTANCE_RELATIONS = {"near", "far"}
DIRECTIONAL = {"north_of", "south_of", "east_of", "west_of"}
ROAD_TOPOLOGY = {"same_side_of_road", "opposite_side_of_road", "across_road",
                 "along_road", "near_intersection"}
ORIENTATION = {"aligned_with", "parallel_to", "perpendicular_to"}


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def classify_parse(record) -> str:
    parsed = record.get("parsed")
    if not record.get("valid") or not parsed:
        return "PARSE_FAILED"
    names = [r.get("relation_normalized")
             for r in (parsed.get("relations") or [])]
    mapped = [n for n in names if n in ONTOLOGY and n not in NON_SPATIAL]
    unknown = [n for n in names if n == "UNKNOWN"]
    if not mapped and not unknown:
        return "NO_SPATIAL_RELATION"
    if mapped and not unknown:
        return "ONTOLOGY_COVERED"
    if mapped and unknown:
        return "PARTIAL_COVERED"
    return "UNKNOWN_ONLY"


def mapped_pairs(record):
    """(relation name, anchor index) for every relation inside the ontology."""
    out = []
    anchors = record.get("anchors") or []
    for rel in (record.get("parsed") or {}).get("relations", []) or []:
        name = rel.get("relation_normalized")
        if name not in ONTOLOGY or name in NON_SPATIAL:
            continue
        index = rel.get("anchor")
        if isinstance(index, str):
            index = next((i for i, a in enumerate(anchors)
                          if (a.get("raw") or {}).get("entity_name") == index),
                         None)
        if not isinstance(index, int) or not (0 <= index < len(anchors)):
            continue
        out.append((name, index))
    return out


def binding_status(record, bound):
    """Where the anchors a mapped relation uses resolve, per §4 and §5.

    Only anchors a *mapped* relation uses are counted: an anchor introduced by an
    UNKNOWN relation plays no part in the score, so calling a sample ambiguous
    because of it would measure the wrong thing.
    """
    used = {index for _, index in mapped_pairs(record)}
    if not used:
        return "BINDING_UNRESOLVED"
    exact = ambiguous = unresolved = 0
    for index in sorted(used):
        nodes = bound.get(index) or []
        if not nodes:
            unresolved += 1
        elif len(nodes) == 1:
            exact += 1
        else:
            ambiguous += 1
    if unresolved:
        return "BINDING_UNRESOLVED"
    return "BINDING_AMBIGUOUS" if ambiguous else "BINDING_CLEAN"


def relation_groups(record) -> set:
    return {name for name, _ in mapped_pairs(record)}


def is_multi_anchor(record) -> bool:
    """Does the sentence need two anchors, or one anchor plus a road factor?"""
    names = relation_groups(record)
    if "between" in names:
        return True
    if names & ROAD_TOPOLOGY:
        return True
    return len({i for _, i in mapped_pairs(record)}) >= 2


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def anchor_nodes(anchor, graph, mode="all"):
    nodes = [graph.by_id[i] for i in
             list(anchor.get("regions", [])) + list(anchor.get("instances", []))
             if i in graph.by_id]
    if mode == "hard" and nodes:
        # Deterministic first hypothesis: regions before instances, and the
        # lowest node id within either.  Deliberately blind to the answer.
        nodes = sorted(nodes, key=lambda n: (n.kind is not NodeKind.ROAD_REGION,
                                             n.node_id))[:1]
    return nodes


def graph_scores(torch, teacher, vocab, graph, candidates, anchors, pairs,
                 rng=None, shuffled=False):
    """Candidate scores marginalised over (relation, anchor) slots."""
    between_idx = sorted({i for n, i in pairs if n == "between"})
    second_for = ({between_idx[0]: between_idx[1], between_idx[1]: between_idx[0]}
                  if len(between_idx) == 2 else {})
    columns = []
    for name, index in pairs:
        if shuffled:
            name = str(rng.choice([r for r in RELATION_NAMES if r != name]))
        nodes = anchors.get(index) or []
        if not nodes:
            continue
        partner = (anchors.get(second_for[index]) or [None])[0] \
            if name == "between" and index in second_for else None
        ctx = EdgeContext(road_regions=graph.regions,
                          road_members=graph.context.road_members,
                          second_anchor=(None if partner is None
                                         else np.asarray(partner.center[:2])))
        geom = np.stack([edge_vector(node, c, ctx)
                         for node in nodes for c in candidates])
        with torch.no_grad():
            out = teacher(torch.as_tensor(geom, dtype=torch.float32),
                          torch.full((len(geom),), vocab.get(name, 0),
                                     dtype=torch.long)).numpy()
        out = out.reshape(len(nodes), len(candidates))
        peak = out.max(axis=0, keepdims=True)
        columns.append((peak + np.log(np.exp(out - peak).sum(axis=0,
                                                             keepdims=True))
                        ).squeeze(0))
    if not columns:
        return None
    stacked = np.stack(columns)
    peak = stacked.max(axis=0)
    return peak + np.log(np.exp(stacked - peak).sum(axis=0))


def prior_scores(graph, candidates, anchors, pairs):
    best = np.full(len(candidates), -np.inf)
    for _, index in pairs:
        for node in anchors.get(index) or []:
            for i, c in enumerate(candidates):
                best[i] = max(best[i], -footprint_distance(node, c))
    return best if np.isfinite(best).any() else None


def oracle_anchors(graph, anchors, target):
    """Diagnostic only: keep the anchor hypothesis nearest the true target.

    Labelled ORACLE wherever it is reported and never mixed into a deployable
    table.  It exists to price the ambiguity -- how much the score would gain if
    the right hypothesis were known -- not to be a method.
    """
    out = {}
    for index, nodes in anchors.items():
        if not nodes:
            out[index] = nodes
            continue
        best = min(nodes, key=lambda n: footprint_distance(n, target))
        out[index] = [best]
    return out


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
    teacher.load_state_dict(torch.load(out_dir / "relation_teacher.pt",
                                       map_location="cpu"))
    teacher.eval()

    objects_by_map = load_landmarks(cfg)
    parses = {json.loads(line)["key"]: json.loads(line)
              for line in (out_dir / "deepseek_parse_val_unseen.jsonl")
              .read_text().splitlines() if line.strip()}
    rows = [json.loads(line) for line in
            (artifact_dir(cfg) / "relation_v2" / "relation_samples.jsonl")
            .read_text().splitlines() if line.strip()]

    graphs, samples = {}, []
    for row in rows:
        if row["split"] != "val_unseen" or row["key"] not in parses:
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
        bound = {i: anchor_nodes(a, graph)
                 for i, a in enumerate(record.get("anchors") or [])}
        samples.append({
            "key": row["key"], "map": row["map"], "graph": graph,
            "record": record, "candidates": candidates, "bound": bound,
            "pairs": mapped_pairs(record),
            "target_index": row["target_index"],
            "parse_class": classify_parse(record),
            "binding_class": binding_status(record, bound),
            "relations": relation_groups(record),
        })

    def target_of(s):
        return s["candidates"][s["target_index"]]

    def run(subset, arm, rng):
        out = []
        for s in subset:
            if arm == "distance":
                scores = prior_scores(s["graph"], s["candidates"], s["bound"],
                                      s["pairs"])
            elif arm == "oracle_anchor":
                # ORACLE ONLY
                oracle = oracle_anchors(s["graph"], s["bound"], target_of(s))
                scores = graph_scores(torch, teacher, vocab, s["graph"],
                                      s["candidates"], oracle, s["pairs"],
                                      rng=rng)
            elif arm == "hard_binding":
                hard = {i: anchor_nodes(a, s["graph"], "hard")
                        for i, a in enumerate(s["record"].get("anchors") or [])}
                scores = graph_scores(torch, teacher, vocab, s["graph"],
                                      s["candidates"], hard, s["pairs"],
                                      rng=rng)
            else:
                scores = graph_scores(torch, teacher, vocab, s["graph"],
                                      s["candidates"], s["bound"], s["pairs"],
                                      rng=rng, shuffled=(arm == "shuffled"))
            if scores is None or not np.isfinite(scores).any():
                continue
            row = rank_metrics(np.asarray(scores, dtype=np.float64),
                               s["target_index"])
            row["key"] = s["key"]
            row["relations"] = sorted(s["relations"])
            out.append(row)
        return out

    def table(subset, arms=("distance", "shuffled", "graph"),
              extra=()):
        """Every arm on the same samples, so the rows are comparable."""
        if not subset:
            return {}, {}, set()
        raw = {a: run(subset, a, np.random.default_rng(0))
               for a in tuple(arms) + tuple(extra)}
        common = set.intersection(*[{r["key"] for r in v} for v in raw.values()])
        rows_by_arm = {a: [r for r in v if r["key"] in common]
                       for a, v in raw.items()}
        return ({a: aggregate(v) for a, v in rows_by_arm.items()},
                rows_by_arm, common)

    covered = [s for s in samples if s["parse_class"] == "ONTOLOGY_COVERED"]
    partial = [s for s in samples if s["parse_class"] == "PARTIAL_COVERED"]
    unknown = [s for s in samples if s["parse_class"] == "UNKNOWN_ONLY"]
    clean = [s for s in covered if s["binding_class"] == "BINDING_CLEAN"]
    ambiguous = [s for s in covered if s["binding_class"] == "BINDING_AMBIGUOUS"]
    unresolved = [s for s in covered if s["binding_class"] == "BINDING_UNRESOLVED"]

    subsets = {
        "ontology_covered": covered,
        "binding_clean": clean,
        "binding_ambiguous": ambiguous,
        "binding_unresolved": unresolved,
        "partial_covered": partial,
        "unknown_only": unknown,
        "non_distance": [s for s in covered
                         if not (s["relations"] & DISTANCE_RELATIONS)],
        "multi_anchor": [s for s in covered if is_multi_anchor(s["record"])],
    }

    report = {"counts": {k: len(v) for k, v in subsets.items()},
              "population": dict(Counter(s["parse_class"] for s in samples)),
              "binding": dict(Counter(s["binding_class"] for s in covered)),
              "tables": {}, "per_relation": {}, "counterfactual": {},
              "ambiguity": {}, "n_samples": len(samples)}

    for key, subset in subsets.items():
        arms, rows_by_arm, common = table(subset)
        if not arms:
            continue
        report["tables"][key] = {"n": len(common),
                                 "aggregate": arms,
                                 "relations": dict(Counter(
                                     r for s in subset
                                     for r in s["relations"]))}
        # Gate E is decided on the covered-and-clean subset.
        if key in ("binding_clean", "ontology_covered") and \
                {"distance", "shuffled", "graph"} <= set(rows_by_arm):
            a = np.array([r["top1"] for r in rows_by_arm["graph"]], dtype=bool)
            for other in ("shuffled", "distance"):
                b_map = {r["key"]: r["top1"] for r in rows_by_arm[other]}
                b = np.array([bool(b_map.get(r["key"], 0))
                              for r in rows_by_arm["graph"]], dtype=bool)
                report.setdefault("gate_e_pairs", {})[f"graph_vs_{other}_{key}"] = \
                    mcnemar(a, b)

    # ---- per relation, on the covered subset
    for name in RELATION_NAMES:
        sub = [s for s in covered if name in s["relations"]]
        arms, rows_by_arm, common = table(sub)
        if not arms or not common:
            continue
        report["per_relation"][name] = {
            "n": len(common),
            "distance": arms["distance"]["top1"],
            "graph": arms["graph"]["top1"],
            "shuffled": arms["shuffled"]["top1"],
            "graph_mrr": arms["graph"]["mrr"],
            "distance_mrr": arms["distance"]["mrr"],
            "graph_margin": arms["graph"]["median_margin"],
            # AUC of the graph's own ranking of the true target against its
            # distractors, which is what a Top-1 cannot show on small subsets.
            "pairwise_auc": float(np.mean([
                pairwise_auc(np.array([row["margin"]]), np.array([0.0]))
                for row in rows_by_arm["graph"]])) if rows_by_arm["graph"] else 0.0,
        }

    # ---- counterfactual, on covered + clean
    if clean:
        pairs = []
        for s in clean:
            names = [n for n, _ in s["pairs"] if n in RELATION_NAMES]
            if not names:
                continue
            rng = np.random.default_rng(0)
            correct = graph_scores(torch, teacher, vocab, s["graph"],
                                   s["candidates"], s["bound"], s["pairs"],
                                   rng=rng)
            opposite_pairs = []
            from sensaturban_fpv.spatial_graph.relation_geometry import OPPOSITE
            for n, i in s["pairs"]:
                other = OPPOSITE.get(n)
                if other is not None:
                    opposite_pairs.append((other, i))
            if not opposite_pairs:
                continue
            opposite = graph_scores(torch, teacher, vocab, s["graph"],
                                    s["candidates"], s["bound"], opposite_pairs,
                                    rng=rng)
            if correct is None or opposite is None:
                continue
            t = s["target_index"]
            pairs.append({"key": s["key"],
                          "correct": float(correct[t]),
                          "opposite": float(opposite[t]),
                          "win": float(correct[t] > opposite[t])})
        if pairs:
            win = np.array([p["win"] for p in pairs], dtype=bool)
            from scipy.stats import binomtest
            wins = int(win.sum())
            report["counterfactual"] = {
                "n": len(pairs),
                "mean_correct": float(np.mean([p["correct"] for p in pairs])),
                "mean_opposite": float(np.mean([p["opposite"] for p in pairs])),
                "win_rate": float(win.mean()),
                "wins": wins, "losses": len(pairs) - wins,
                "p_value": float(binomtest(wins, len(pairs), 0.5).pvalue),
            }

    # ---- ambiguity: hard vs marginalised vs oracle, on the ambiguous subset
    if ambiguous:
        arms, rows_by_arm, common = table(
            ambiguous, arms=(), extra=("hard_binding", "graph", "oracle_anchor"))
        if arms:
            report["ambiguity"] = {"n": len(common),
                                   "variants": arms}

    (out_dir / "ontology_subset_gate.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")

    print(f"samples {len(samples)}")
    print(f"population {report['population']}")
    print(f"binding within covered {report['binding']}")
    print(f"subsets {report['counts']}\n")
    for key in ("ontology_covered", "binding_clean", "binding_ambiguous",
                "non_distance", "multi_anchor", "partial_covered",
                "unknown_only"):
        entry = report["tables"].get(key)
        if not entry:
            continue
        agg = entry["aggregate"]
        parts = "  ".join(
            f"{a}={agg[a]['top1']:.4f}" if agg[a].get("n") else f"{a}=--"
            for a in ("distance", "shuffled", "graph") if a in agg)
        print(f"{key:20s} n={entry['n']:4d}  {parts}")
    if report["counterfactual"]:
        c = report["counterfactual"]
        print(f"\ncounterfactual n={c['n']} win={c['win_rate']:.3f} "
              f"p={c['p_value']:.2e} correct={c['mean_correct']:.2f} "
              f"opposite={c['mean_opposite']:.2f}")
    print(f"-> {out_dir / 'ontology_subset_gate.json'}", flush=True)


if __name__ == "__main__":
    main()
