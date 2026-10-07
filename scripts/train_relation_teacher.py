#!/usr/bin/env python3
"""Phase 3: train the teacher on clean supervision, then run Gate A on it.

Gate A asks one thing: does the model score the *true* relation above a wrong
one, on the same geometry?  Everything else here is in service of answering that
question honestly.

* The test split is four maps that appear in no training or threshold-selection
  set, so the teacher cannot have memorised their geometry.
* Every control is checked to have moved the scores before its numbers are
  reported.  A control that silently no-ops reads as a pass, and two earlier
  attempts at an anchor-shuffle control in this project were exactly that.
* Hyper-parameters are chosen on synthetic-val, never on test.

If Gate A fails there is no point continuing: the labels are geometry, so a
model that cannot fit them is failing at something that is not the corpus's
fault, and Phase 4 would only add a second possible cause to the same failure.
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

from sensaturban_fpv.config import artifact_dir, load_config  # noqa: E402
from sensaturban_fpv.spatial_graph.edge_features import EDGE_DIM  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_geometry import RELATION_NAMES  # noqa: E402
from sensaturban_fpv.spatial_graph.relation_teacher import (  # noqa: E402
    aggregate, arm_relations, build_model, count_params, mcnemar, pairwise_auc,
    rank_metrics, relation_vocab,
)

ARMS = ("correct", "opposite", "shuffled_relation", "zero_relation",
        "shuffled_anchor")

# Filled in by main(); module level so the scoring helper does not depend on
# import order.
ARM_VOCAB = {}


def load_examples(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


class Tensors:
    """Positives and matched negatives padded to one width."""

    def __init__(self, rows, vocab, max_neg=8):
        n = len(rows)
        self.edge = np.zeros((n, EDGE_DIM), dtype=np.float32)
        self.neg = np.zeros((n, max_neg, EDGE_DIM), dtype=np.float32)
        self.neg_valid = np.zeros((n, max_neg), dtype=bool)
        self.rel = np.zeros(n, dtype=np.int64)
        self.keys = []
        self.relations = []
        for i, row in enumerate(rows):
            self.edge[i] = np.asarray(row["positive_edge"], dtype=np.float32)
            negs = row["negative_edges"][:max_neg]
            self.neg[i, :len(negs)] = np.asarray(negs, dtype=np.float32)
            self.neg_valid[i, :len(negs)] = True
            self.rel[i] = vocab[row["relation"]]
            self.keys.append((row["map"], row["anchor_id"], row["positive_id"]))
            self.relations.append(row["relation"])


def to_torch(torch, tensors, device):
    return {
        "edge": torch.as_tensor(tensors.edge, dtype=torch.float32, device=device),
        "neg": torch.as_tensor(tensors.neg, dtype=torch.float32, device=device),
        "neg_valid": torch.as_tensor(tensors.neg_valid, dtype=torch.bool,
                                     device=device),
        "rel": torch.as_tensor(tensors.rel, dtype=torch.long, device=device),
        "keys": tensors.keys, "relations": tensors.relations,
    }


def score_arms(torch, model, batch, arm, donor=None, chunk=512):
    """Positive and negative scores under one arm's relation assignment."""
    edge, neg = batch["edge"], batch["neg"]
    n = edge.shape[0]
    rel = batch["rel"]
    if arm == "shuffled_anchor" and donor is not None:
        # The whole geometry -- positive and negatives -- comes from a different
        # example, while this example's relation is kept.  The edge vector
        # encodes the anchor implicitly, so donating geometry *is* the anchor
        # shuffle; there is no anchor slot to permute.
        edge = edge[donor]
        neg = neg[donor]
        rels = rel.cpu().numpy()
    else:
        rels = arm_relations(arm, rel.cpu().numpy(), ARM_VOCAB, seed=0,
                             rng=np.random.default_rng(0))
    rel_t = torch.as_tensor(rels, dtype=torch.long, device=edge.device)
    pos_out = np.zeros(n, dtype=np.float64)
    neg_out = np.zeros((n, neg.shape[1]), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            pos_out[start:stop] = model(edge[start:stop], rel_t[start:stop]).cpu().numpy()
            flat = model(neg[start:stop].reshape(-1, EDGE_DIM),
                         rel_t[start:stop, None].expand(-1, neg.shape[1]).reshape(-1))
            neg_out[start:stop] = flat.reshape(stop - start, -1).cpu().numpy()
    return pos_out, neg_out


def evaluate(torch, model, batch, arm, donor=None):
    pos, neg = score_arms(torch, model, batch, arm, donor)
    valid = batch["neg_valid"].cpu().numpy()
    rows = []
    for i in range(pos.shape[0]):
        scores = np.concatenate([[pos[i]], neg[i][valid[i]]])
        row = rank_metrics(scores)
        row["relation"] = batch["relations"][i]
        rows.append(row)
    return rows, pos, neg, valid


def main() -> None:
    global ARM_VOCAB
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    # 249k examples against a 44k-parameter model: one pass is already a lot of
    # updates, and the early-stopping behaviour is visible in the tuning table.
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--tune-epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else artifact_dir(cfg) / "spatial_graph"
    import torch
    torch.manual_seed(args.seed)
    # This box has 12 cores and other work on it.  Left to itself torch takes
    # one thread per core and then thrashes: the first attempt burned 6400
    # CPU-seconds without finishing ten epochs, because 17 threads were
    # competing with a concurrent job for the same cores.  Four threads is
    # faster here in wall-clock than sixteen.
    torch.set_num_threads(4)
    device = torch.device("cpu")

    vocab = relation_vocab(RELATION_NAMES)
    ARM_VOCAB = vocab
    data = {s: Tensors(load_examples(out_dir / f"synthetic_relations_{s}.jsonl"),
                       vocab)
            for s in ("train", "val", "test")}
    print(f"examples: train {len(data['train'].rel)} "
          f"val {len(data['val'].rel)} test {len(data['test'].rel)}", flush=True)
    print(f"relations present: {sorted(Counter(data['train'].relations).items())}",
          flush=True)

    def run(epochs, lr, weight_decay, hidden):
        torch.manual_seed(args.seed)
        model = build_model(torch, len(vocab), hidden=hidden)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        batch = to_torch(torch, data["train"], device)
        edge, neg, valid, rel = (batch["edge"], batch["neg"], batch["neg_valid"],
                                 batch["rel"])
        n = edge.shape[0]
        rng = np.random.default_rng(args.seed)
        for epoch in range(epochs):
            order = torch.as_tensor(rng.permutation(n))
            total, steps = 0.0, 0
            for start in range(0, n, 256):
                idx = order[start:start + 256]
                pos = model(edge[idx], rel[idx])
                negs = model(neg[idx].reshape(-1, EDGE_DIM),
                             rel[idx, None].expand(-1, neg.shape[1]).reshape(-1)
                             ).reshape(idx.shape[0], -1)
                negs = negs.masked_fill(~valid[idx], -1e9)
                logits = torch.cat([pos[:, None], negs], dim=1)
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.zeros(idx.shape[0], dtype=torch.long))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach())
                steps += 1
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"    epoch {epoch:3d} loss {total / max(steps,1):.4f}",
                      flush=True)
        return model

    # ---- hyper-parameters on synthetic-val
    tuning = []
    best = None
    for lr in (3e-3, 1e-2):
        for hidden in (128,):
            for wd in (1e-4,):
                model = run(args.tune_epochs, lr, wd, hidden)
                rows, _, _, _ = evaluate(torch, model,
                                         to_torch(torch, data["val"], device),
                                         "correct")
                top1 = aggregate(rows)["top1"]
                tuning.append({"lr": lr, "hidden": hidden, "weight_decay": wd,
                               "val_top1": top1})
                print(f"  lr={lr} hidden={hidden} wd={wd} "
                      f"val top1 {top1:.4f}", flush=True)
                if best is None or top1 > best[0]:
                    best = (top1, lr, hidden, wd)
    _, lr, hidden, wd = best
    print(f"  chosen lr={lr} hidden={hidden}", flush=True)

    model = run(args.epochs, lr, wd, hidden)
    params = count_params(model)
    # Saved so the grounding stage reads the teacher that the gate has just
    # judged, rather than refitting one and hoping it is the same.
    torch.save(model.state_dict(), out_dir / "relation_teacher.pt")

    # ---- Gate A on the held-out maps
    batch = to_torch(torch, data["test"], device)
    n = batch["edge"].shape[0]
    rng = np.random.default_rng(args.seed + 1)
    donor = rng.permutation(n)
    for i in np.flatnonzero(donor == np.arange(n)):
        j = (i + 1) % n
        donor[i], donor[j] = donor[j], donor[i]

    results, per_relation = {}, {}
    scores_by_arm = {}
    for arm in ARMS:
        rows, pos, neg, valid = evaluate(
            torch, model, batch, arm,
            torch.as_tensor(donor, dtype=torch.long) if arm == "shuffled_anchor"
            else None)
        results[arm] = aggregate(rows)
        scores_by_arm[arm] = (pos, neg, valid, rows)
        per_relation[arm] = {
            r: aggregate([x for x in rows if x["relation"] == r])
            for r in sorted({x["relation"] for x in rows})}

    # ---- prove the controls moved the scores
    moved = {"shuffled_anchor": float(np.abs(
        scores_by_arm["correct"][0] - scores_by_arm["shuffled_anchor"][0]).max())}
    for arm in ("opposite", "shuffled_relation", "zero_relation"):
        moved[arm] = float(np.abs(
            scores_by_arm["correct"][0] - scores_by_arm[arm][0]).max())
    print(f"  control score movement: {json.dumps(moved)}", flush=True)
    for arm, value in moved.items():
        if arm != "shuffled_anchor" and value < 1e-6:
            raise SystemExit(f"control {arm} is a no-op; refusing to report it")

    # ---- pairwise AUC, on the ranking and on the counterfactual
    pos_c, neg_c, valid, _ = scores_by_arm["correct"]
    auc_ranking = float(np.nanmean([
        pairwise_auc(pos_c[i:i + 1], neg_c[i][valid[i]]) for i in range(n)]))
    auc_counter = float(np.nanmean([
        pairwise_auc(pos_c[i:i + 1],
                     scores_by_arm["opposite"][0][i:i + 1]) for i in range(n)]))

    correct_top1 = np.array([r["top1"] for r in scores_by_arm["correct"][3]],
                            dtype=bool)
    gate = {"correct_top1": results["correct"]["top1"],
            "shuffled_relation_top1": results["shuffled_relation"]["top1"],
            "opposite_top1": results["opposite"]["top1"],
            "zero_relation_top1": results["zero_relation"]["top1"],
            "shuffled_anchor_top1": results["shuffled_anchor"]["top1"],
            "pairwise_auc_ranking": auc_ranking,
            "pairwise_auc_counterfactual": auc_counter,
            "control_score_movement": moved,
            "chosen": {"lr": lr, "hidden": hidden, "epochs": args.epochs}}
    for arm in ARMS[1:]:
        other = np.array([r["top1"] for r in scores_by_arm[arm][3]], dtype=bool)
        gate[f"correct_vs_{arm}"] = mcnemar(correct_top1, other)

    conditions = {
        "A1_correct_beats_shuffled_relation": (
            gate["correct_top1"] > gate["shuffled_relation_top1"]
            and gate["correct_vs_shuffled_relation"]["p_value"] < 0.05),
        "A2_correct_beats_opposite_relation": (
            gate["correct_top1"] > gate["opposite_top1"]
            and gate["correct_vs_opposite"]["p_value"] < 0.05),
        "A3_zeroing_the_relation_hurts": (
            gate["correct_top1"] - gate["zero_relation_top1"] > 0.10),
        "A4_shuffling_the_anchor_hurts": (
            gate["correct_top1"] - gate["shuffled_anchor_top1"] > 0.10),
        "A5_counterfactual_auc_above_half": auc_counter > 0.5,
    }
    gate["conditions"] = conditions
    gate["passed"] = all(conditions.values())

    report = {"params": {"teacher": params}, "tuning": tuning,
              "results": results, "per_relation": per_relation, "gate": gate,
              "n_test": int(n),
              "relations": sorted(set(batch["relations"]))}
    (out_dir / "synthetic_teacher_gate.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")

    print(f"\n{'arm':22s} {'top1':>7s} {'top4':>7s} {'mrr':>7s} {'margin':>8s}")
    for arm in ARMS:
        r = results[arm]
        print(f"{arm:22s} {r['top1']:7.4f} {r['top4']:7.4f} {r['mrr']:7.4f} "
              f"{r['median_margin']:8.3f}")
    print(f"  pairwise AUC (ranking)       {auc_ranking:.4f}")
    print(f"  pairwise AUC (counterfactual) {auc_counter:.4f}")
    print(json.dumps(conditions, indent=1), flush=True)
    print(f"GATE A {'PASS' if gate['passed'] else 'FAIL'}", flush=True)
    print(f"-> {out_dir / 'synthetic_teacher_gate.json'}", flush=True)


if __name__ == "__main__":
    main()
