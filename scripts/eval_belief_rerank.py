#!/usr/bin/env python3
"""Rerank the top-down shortlist with a belief over relations, frames and anchors.

The question is not whether the graph can ground on its own -- five rounds have
answered that, and the answer is no.  It is whether a *distribution* over
readings does better than a commitment, on the candidates the visual channel has
already narrowed to four.  ``TD_masked`` puts the target in its top four 75.7% of
the time and first 32.0% of the time; the gap between those two numbers is the
only headroom that matters here.

Scoring, per candidate::

    S = S_TD + lambda_ref * z(S_reference) + lambda_rel * z(S_belief)

where ``S_belief`` marginalises over the language's hypotheses *and* over the map
entities each anchor phrase could denote::

    S_belief(c) = logsumexp_{h, j} [ log P(h) + log P(j | phrase_h)
                                     + geometry(c | h, j) ]

``geometry`` is the frozen teacher for the fifteen unambiguous relations and an
explicit frame rule for ``behind``/``front``/``left``/``right``, whose frames are
exactly what the language is uncertain about.  The teacher is not retrained; the
frames are carried as competing hypotheses rather than trained in.

``lambda_ref`` and ``lambda_rel`` are chosen on ``val_seen`` only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.config import artifact_dir, load_config, load_landmarks  # noqa: E402
from sensaturban_fpv.spatial_graph import build_block_graph  # noqa: E402
from sensaturban_fpv.spatial_graph.edge_features import (  # noqa: E402
    EdgeContext, edge_vector, footprint_distance,
)
from sensaturban_fpv.spatial_graph.node_types import NodeKind  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_belief import (  # noqa: E402
    FRAME_RELATIONS, TEACHER_RELATIONS, frame_axis, frame_score, marginalise,
)
from sensaturban_fpv.spatial_graph.relation_geometry import RELATION_NAMES  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_teacher import (  # noqa: E402
    aggregate, build_model, mcnemar, rank_metrics, relation_vocab,
)

SPLITS = ("val_seen", "val_unseen")
GROUPS = ("building", "vehicle", "other")


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    s = v.std()
    return np.zeros_like(v) if s < 1e-9 else (v - v.mean()) / s


def load_td(cfg, base: Path, keys) -> dict:
    """The frozen masked top-down score, in candidate order."""
    out = {}
    for key in keys:
        path = base / key
        if not path.exists():
            continue
        archive = np.load(path, allow_pickle=True)
        phrase = (archive["text_phrase"] if "text_phrase" in archive.files
                  else None)
        if phrase is None or not np.asarray(phrase).size:
            continue
        out[key] = {
            "td": np.asarray(archive["td_context_masked"], np.float32)
            @ np.asarray(phrase, np.float32),
            "types": [str(t) for t in archive["entity_types"]],
        }
    return out


class Bound:
    """Everything needed to score one sample under one arm."""

    def __init__(self, row, record, graph, td, cfg):
        self.key = row["key"]
        self.split = row["split"]
        self.map = row["map"]
        self.target_index = row["target_index"]
        self.graph = graph
        self.candidates = [graph.by_id[c] for c in row["candidate_ids"]
                           if c in graph.by_id]
        self.uav_xy = np.asarray(row["uav_xy"], dtype=np.float64)
        self.uav_yaw = float(row["uav_yaw"])
        self.anchors = [self._bind(a) for a in record.get("anchors", [])]
        self.hypotheses = record.get("hypotheses", [])
        self.confidence = float(record.get("confidence", 0.0))
        self.td = td
        self._rel_cache = {}
        self.group = ("building"
                      if td is not None and td["types"][self.target_index] == "Building"
                      else "vehicle"
                      if td is not None and td["types"][self.target_index]
                      in ("Car", "Bike") else "other")

    def _bind(self, anchor):
        nodes = [self.graph.by_id[i] for i in
                 list(anchor.get("regions", [])) + list(anchor.get("instances", []))
                 if i in self.graph.by_id]
        return nodes

    def nearest_road(self, node):
        """(region, members) of the road region nearest ``node``.

        The region rather than its frame: a frame-dependent word read against
        the road needs the road's own tangent, and the per-node cache in
        ``BlockGraph.road_frame_for`` keeps only the frame.
        """
        from sensaturban_fpv.spatial_graph.edge_features import road_frame
        best, best_d = None, None
        for idx, region in enumerate(self.graph.regions):
            members = self.graph.context.road_members.get(idx, [])
            if not members:
                continue
            frame = road_frame(node, region, members)
            if best_d is None or frame.distance < best_d:
                best, best_d = (region, tuple(members)), frame.distance
        return best

    def context(self, anchor, second=None):
        return EdgeContext(
            road_regions=self.graph.regions,
            road_members=self.graph.context.road_members,
            second_anchor=(None if second is None
                           else np.asarray(second.center[:2])),
            cross_road=None)


def geometry_scores(torch, teacher, vocab, sample, hypothesis, nodes,
                    second=None):
    """``geometry(candidate | hypothesis, anchor node)`` for every candidate."""
    relation = hypothesis["relation"]
    frame = hypothesis["frame"]
    ctx = sample.context(nodes[0], second)
    if relation in TEACHER_RELATIONS:
        geom = np.stack([edge_vector(node, c, ctx) for node in nodes
                         for c in sample.candidates])
        with torch.no_grad():
            out = teacher(torch.as_tensor(geom, dtype=torch.float32),
                          torch.full((len(geom),), vocab.get(relation, 0),
                                     dtype=torch.long)).numpy()
        out = out.reshape(len(nodes), len(sample.candidates))
        peak = out.max(axis=0, keepdims=True)
        return (peak + np.log(np.exp(out - peak).sum(axis=0, keepdims=True))
                ).squeeze(0)
    if relation in FRAME_RELATIONS:
        stacked = []
        for node in nodes:
            frame_ctx = _FrameCtx(sample.uav_yaw, sample.nearest_road(node))
            stacked.append(np.array([
                frame_score(relation, *frame_axis(frame, node, c, frame_ctx))
                for c in sample.candidates], dtype=np.float64))
        out = np.stack(stacked)
        peak = out.max(axis=0, keepdims=True)
        return (peak + np.log(np.exp(out - peak).sum(axis=0, keepdims=True))
                ).squeeze(0)
    return None


class _FrameCtx:
    """The little bit of context ``frame_axis`` needs, in one object."""

    def __init__(self, yaw, road):
        self.agent_yaw = yaw
        self.road, self.road_members = road if road else (None, ())


def belief_scores(torch, teacher, vocab, sample, hypotheses=None, top_k=5,
                  entropy_gate=True, shuffle_relations=False, shuffle_anchors=False,
                  rng=None):
    """``S_belief`` for one sample, marginalised over hypotheses and anchors."""
    hypotheses = hypotheses if hypotheses is not None else sample.hypotheses
    if not hypotheses:
        return None
    hypotheses = hypotheses[:top_k]
    columns = []
    for h in hypotheses:
        relation = h["relation"]
        if shuffle_relations:
            relation = str(rng.choice([r for r in RELATION_NAMES
                                       if r not in TEACHER_RELATIONS]
                                      + list(TEACHER_RELATIONS)))
        anchor_ids = h.get("anchors") or []
        if shuffle_anchors:
            pool = [j for j, a in enumerate(sample.anchors) if a]
            if pool:
                anchor_ids = [int(rng.choice(pool))]
        weight = np.log(max(h["probability"], 1e-9))
        if entropy_gate:
            weight += np.log(max(sample.confidence, 1e-3))
        for anchor_id in anchor_ids:
            if not (0 <= anchor_id < len(sample.anchors)):
                continue
            nodes = sample.anchors[anchor_id]
            if not nodes:
                continue
            second = None
            if relation == "between" and len(anchor_ids) == 2:
                other = [a for a in anchor_ids if a != anchor_id]
                if other and 0 <= other[0] < len(sample.anchors):
                    pool = sample.anchors[other[0]]
                    second = pool[0] if pool else None
            scores = geometry_scores(torch, teacher, vocab, sample,
                                     {"relation": relation, "frame": h["frame"]},
                                     nodes, second)
            if scores is None:
                continue
            # P(anchor node | phrase) is uniform over the bound nodes: nothing
            # in the sentence, and nothing available at test time, says which of
            # several same-named entities is meant.
            columns.append(scores + weight - np.log(len(nodes)))
    if not columns:
        return None
    stacked = np.stack(columns)
    return np.array([marginalise([0.0] * stacked.shape[0], stacked[:, c])
                     for c in range(stacked.shape[1])])


def reference_prior(sample):
    """Distance to the bound anchors: the fallback when a relation is unclear."""
    best = np.full(len(sample.candidates), -np.inf)
    for nodes in sample.anchors:
        for node in nodes:
            for i, c in enumerate(sample.candidates):
                best[i] = max(best[i], -footprint_distance(node, c))
    return best if np.isfinite(best).any() else None


def evaluate(sample, scores):
    if scores is None or not np.isfinite(scores).any():
        return None
    row = rank_metrics(np.asarray(scores, dtype=np.float64), sample.target_index)
    row["key"] = sample.key
    row["group"] = sample.group
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0, 2.0])
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
    rows = [json.loads(line) for line in
            (artifact_dir(cfg) / "relation_v2" / "relation_samples.jsonl")
            .read_text().splitlines() if line.strip()]
    parses = {}
    for split in SPLITS:
        path = out_dir / f"belief_parse_{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                parses[record["key"]] = record
    print(f"parses loaded: {len(parses)}", flush=True)

    visual_base = artifact_dir(cfg) / "entity_grounding" / "data"
    keys = [r["key"] for r in rows if r["key"] in parses]
    td_scores = load_td(cfg, visual_base, keys)
    print(f"top-down scores available for {len(td_scores)}", flush=True)

    graphs, samples = {}, {s: [] for s in SPLITS}
    for row in rows:
        record = parses.get(row["key"])
        if record is None or not record.get("hypotheses"):
            continue
        if row["map"] not in graphs:
            graphs[row["map"]] = build_block_graph(objects_by_map[row["map"]],
                                                   row["map"])
        td = td_scores.get(row["key"])
        sample = Bound(row, record, graphs[row["map"]], td, cfg)
        if len(sample.candidates) != len(row["candidate_ids"]):
            continue
        samples[row["split"]].append(sample)
    print("samples: " + ", ".join(f"{s} {len(v)}" for s, v in samples.items()),
          flush=True)

    # ---- cache the two score components once
    def components(sample, **kw):
        if sample._rel_cache is None:
            sample._rel_cache = {}
        key = (kw.get("top_k", 5), kw.get("entropy_gate", True),
               kw.get("shuffle_relations", False), kw.get("shuffle_anchors", False))
        if key not in sample._rel_cache:
            sample._rel_cache[key] = belief_scores(
                torch, teacher, vocab, sample, top_k=key[0],
                entropy_gate=key[1], shuffle_relations=key[2],
                shuffle_anchors=key[3], rng=np.random.default_rng(0))
        return sample._rel_cache[key]

    def arm_rows(split, lam_ref, lam_rel, **kw):
        out = []
        for sample in samples[split]:
            if sample.td is None:
                continue
            base = zscore(sample.td)
            if lam_ref:
                ref = reference_prior(sample)
                if ref is not None:
                    base = base + lam_ref * zscore(ref)
            if lam_rel:
                rel = components(sample, **kw)
                if rel is not None:
                    base = base + lam_rel * zscore(rel)
            row = evaluate(sample, base)
            if row is not None:
                out.append(row)
        return out

    # ---- lambdas on val_seen
    tuning = []
    for lam_ref in args.lambdas:
        for lam_rel in args.lambdas:
            rows_seen = arm_rows("val_seen", lam_ref, lam_rel)
            if not rows_seen:
                continue
            tuning.append({"lambda_ref": lam_ref, "lambda_rel": lam_rel,
                           "val_seen_top1": aggregate(rows_seen)["top1"]})
    base_seen = aggregate(arm_rows("val_seen", 0.0, 0.0))["top1"]
    best = max(tuning, key=lambda r: r["val_seen_top1"]) if tuning else \
        {"lambda_ref": 0.0, "lambda_rel": 0.0}
    print(f"val_seen TD-only {base_seen:.4f}; chosen "
          f"lambda_ref={best['lambda_ref']} lambda_rel={best['lambda_rel']} "
          f"({best['val_seen_top1']:.4f})", flush=True)

    arms = {
        "R0_TD_only": (0.0, 0.0, {}),
        "R1_TD_reference": (best["lambda_ref"], 0.0, {}),
        "R2_TD_hard_relation": (best["lambda_ref"], best["lambda_rel"],
                                {"top_k": 1, "entropy_gate": False}),
        "R3_TD_prob_relation": (best["lambda_ref"], best["lambda_rel"],
                                {"entropy_gate": False}),
        "R4_TD_prob_entropy_gate": (best["lambda_ref"], best["lambda_rel"],
                                    {"entropy_gate": True}),
        "R5_TD_ref_prob_gate": (best["lambda_ref"], best["lambda_rel"],
                                {"entropy_gate": True}),
        "R6_shuffled_relation": (best["lambda_ref"], best["lambda_rel"],
                                 {"entropy_gate": True,
                                  "shuffle_relations": True}),
        "R7_shuffled_anchor": (best["lambda_ref"], best["lambda_rel"],
                               {"entropy_gate": True, "shuffle_anchors": True}),
    }
    results, per_arm_rows = {}, {}
    for name, (lam_ref, lam_rel, kw) in arms.items():
        per_arm_rows[name] = {s: arm_rows(s, lam_ref, lam_rel, **kw)
                              for s in SPLITS}
        results[name] = {s: aggregate(per_arm_rows[name][s]) for s in SPLITS}

    # ---- ablations
    k_ablation = {}
    for k in (1, 3, 5, 10):
        k_ablation[k] = aggregate(arm_rows("val_unseen", best["lambda_ref"],
                                           best["lambda_rel"], top_k=k,
                                           entropy_gate=True))

    report = {
        "chosen": {"lambda_ref": best["lambda_ref"],
                   "lambda_rel": best["lambda_rel"],
                   "val_seen_td_only": base_seen},
        "tuning": tuning,
        "results": results,
        "k_ablation": k_ablation,
        "n": {s: len(v) for s, v in samples.items()},
        "n_with_td": {s: int(sum(1 for x in v if x.td is not None))
                      for s, v in samples.items()},
    }

    # ---- per group, per relation, rescue
    for name in ("R0_TD_only", "R5_TD_ref_prob_gate"):
        report.setdefault("per_group", {})[name] = {
            g: aggregate([r for r in per_arm_rows[name]["val_unseen"]
                          if r["group"] == g]) for g in GROUPS}
    base_map = {r["key"]: r for r in per_arm_rows["R0_TD_only"]["val_unseen"]}
    fin_map = {r["key"]: r for r in per_arm_rows["R5_TD_ref_prob_gate"]["val_unseen"]}
    rescue = {"n_top4_not_top1": 0, "rescued": 0, "regressed": 0}
    for key, b in base_map.items():
        f = fin_map.get(key)
        if f is None:
            continue
        if not b["top1"] and b["top4"]:
            rescue["n_top4_not_top1"] += 1
            rescue["rescued"] += int(bool(f["top1"]))
        if b["top1"] and not f["top1"]:
            rescue["regressed"] += 1
    if rescue["n_top4_not_top1"]:
        rescue["rescue_rate"] = rescue["rescued"] / rescue["n_top4_not_top1"]
    n_base_correct = sum(1 for b in base_map.values() if b["top1"])
    rescue["regression_rate"] = (rescue["regressed"] / max(n_base_correct, 1))
    report["rescue"] = rescue

    (out_dir / "belief_rerank.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")

    print(f"\n{'arm':28s} {'val_seen':>9s} {'unseen':>9s} {'top4':>7s} {'mrr':>7s} "
          f"{'margin':>8s}")
    for name, r in results.items():
        u = r["val_unseen"]
        vs = r["val_seen"]
        print(f"{name:28s} {vs.get('top1', float('nan')):9.4f} "
              f"{u.get('top1', float('nan')):9.4f} {u.get('top4', float('nan')):7.4f} "
              f"{u.get('mrr', float('nan')):7.4f} "
              f"{u.get('median_margin', float('nan')):8.4f}")
    print(f"\nK ablation (val_unseen): "
          + "  ".join(f"K={k}:{v.get('top1', float('nan')):.4f}"
                      for k, v in k_ablation.items()))
    print(f"rescue {json.dumps(rescue)}")
    print(f"-> {out_dir / 'belief_rerank.json'}", flush=True)


if __name__ == "__main__":
    main()
