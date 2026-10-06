#!/usr/bin/env python3
"""Dual-view candidate grounding: can top-down and oblique45 be fused?

Everything the fusion needs is chosen on ``train_seen`` and ``val_seen``; the
``val_unseen`` manifest is scored once, at the end, with those choices frozen.
That is enforced structurally: :func:`select_on_validation` only ever sees the
validation splits, and the test split is touched only inside :func:`final_eval`.

Fusion happens at the score level on the candidate set the two views share.  A
90-degree oblique does not contain every candidate the orthophoto does, so the
comparison is always on the intersection; fusing over different candidate sets
would not be a fusion.

Score calibration is fitted per view on the validation splits only.  The two
views produce cosine similarities from the *same* frozen encoder, so they are
already on one scale -- calibration is tested rather than assumed, because the
score *distributions* differ even when the scale does not.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from math import comb
from pathlib import Path

import numpy as np

VIEWS_TD = ("td_010", "td_015", "td_020", "td_030")
VIEW_O = "o2048"
VARIANTS = ("phrase", "name")

APPEARANCE_TOKENS = {
    "color": ["red", "white", "black", "gray", "grey", "blue", "green", "brown",
              "yellow", "orange", "silver", "golden", "pink", "purple"],
    "facade": ["striped", "brick", "glass", "facade", "roof", "tower",
               "single", "storey", "storeys", "multi", "storeyed",
               "triangular", "rectangular", "round", "shaped", "arched", "domed"],
    "instance": ["car", "parking", "field", "court", "cycle", "bike", "boat",
                 "bus", "van", "truck", "chimney", "entrance", "door", "window",
                 "fence", "gate", "lamp", "bench", "bin", "sign"],
}
GEOMETRY_TOKENS = ["long", "wide", "narrow", "large", "small", "left", "right",
                   "behind", "front", "near", "beside", "next", "road",
                   "intersection", "corner", "block", "side", "end", "section",
                   "tall", "short", "high", "low"]


def tokens(text: str) -> set:
    out = set()
    for piece in re.split(r"[^a-z0-9]+", (text or "").lower()):
        if not piece:
            continue
        out.add(piece)
        if piece.endswith("s") and len(piece) > 3:
            out.add(piece[:-1])
    return out


def token_flags(phrase: str) -> dict:
    toks = tokens(phrase)
    app = {k: sorted(set(v) & toks) for k, v in APPEARANCE_TOKENS.items()}
    app = {k: v for k, v in app.items() if v}
    geo = sorted(set(GEOMETRY_TOKENS) & toks)
    return {"appearance": app, "geometry": geo,
            "has_appearance": bool(app), "has_geometry": bool(geo)}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def rank_of(sims, ids, target_id):
    if target_id not in ids or len(ids) < 2:
        return None
    i = ids.index(target_id)
    s = np.asarray(sims, dtype=np.float64)
    order = np.argsort(-s)
    rank = int(np.where(order == i)[0][0]) + 1
    others = [s[j] for j in range(len(s)) if j != i]
    best = max(others) if others else float("nan")
    return {"rank": rank, "top1": rank == 1, "top4": rank <= 4, "mrr": 1.0 / rank,
            "margin": float(s[i] - best), "target_sim": float(s[i]),
            "hardest_negative_rank": int(np.argmax(others)) if others else -1}


def agg(metrics):
    m = [x for x in metrics if x]
    if not m:
        return {}
    margins = np.array([x["margin"] for x in m], dtype=np.float64)
    return {
        "n": len(m),
        "top1": float(np.mean([x["top1"] for x in m])),
        "top4": float(np.mean([x["top4"] for x in m])),
        "mrr": float(np.mean([x["mrr"] for x in m])),
        "margin_mean": float(margins.mean()),
        "margin_median": float(np.median(margins)),
        "positive_margin_ratio": float((margins > 0).mean()),
    }


def mcnemar(gain, loss):
    n = gain + loss
    if n == 0:
        return 1.0
    k = min(gain, loss)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)))


def paired(a_metrics, b_metrics):
    pairs = [(x, y) for x, y in zip(a_metrics, b_metrics) if x and y]
    if not pairs:
        return {}
    gain = sum(1 for x, y in pairs if not x["top1"] and y["top1"])
    loss = sum(1 for x, y in pairs if x["top1"] and not y["top1"])
    both_s = sum(1 for x, y in pairs if x["top1"] and y["top1"])
    both_f = sum(1 for x, y in pairs if not x["top1"] and not y["top1"])
    dm = np.array([y["margin"] - x["margin"] for x, y in pairs])
    return {"n": len(pairs), "gain": gain, "loss": loss,
            "both_success": both_s, "both_fail": both_f,
            "delta_top1": float(np.mean([y["top1"] for _, y in pairs])
                                 - np.mean([x["top1"] for x, _ in pairs])),
            "delta_margin_median": float(np.median(dm)),
            "mcnemar_p": mcnemar(gain, loss)}


# --------------------------------------------------------------------------
# sample plumbing
# --------------------------------------------------------------------------

def load_split(root: Path, split: str) -> list:
    records, seen = [], set()
    for path in sorted(root.glob(f"{split}*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["episode_index"], rec["step"])
            if key in seen:
                continue
            seen.add(key)
            records.append(rec)
    return records


def common_view(rec, td_view, o_view, variant):
    """The candidate set both views share, with both score vectors on it."""
    a, b = rec["views"].get(td_view), rec["views"].get(o_view)
    if not a or not b:
        return None
    if variant not in a.get("sims", {}) or variant not in b.get("sims", {}):
        return None
    common = [c for c in a["ids"] if c in set(b["ids"])]
    if rec["target_id"] not in common or len(common) < 2:
        return None
    return {
        "ids": common,
        "s_td": np.array([a["sims"][variant][a["ids"].index(c)] for c in common]),
        "s_o": np.array([b["sims"][variant][b["ids"].index(c)] for c in common]),
        "target_id": rec["target_id"],
    }


def scores_for(records, td_view, variant, calibrator=None):
    """Per record: the shared ids plus calibrated td/oblique scores."""
    out = []
    for rec in records:
        cv = common_view(rec, td_view, VIEW_O, variant)
        if cv is None:
            out.append(None)
            continue
        s_td, s_o = cv["s_td"], cv["s_o"]
        if calibrator:
            s_td = calibrator["td"].apply(s_td)
            s_o = calibrator["o"].apply(s_o)
        out.append({"rec": rec, "ids": cv["ids"], "target_id": cv["target_id"],
                    "s_td": s_td, "s_o": s_o})
    return out


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

class ZScore:
    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, max(sigma, 1e-9)

    def apply(self, s):
        return (np.asarray(s, dtype=np.float64) - self.mu) / self.sigma


class Temperature:
    def __init__(self, T):
        self.T = max(float(T), 1e-3)

    def apply(self, s):
        return np.asarray(s, dtype=np.float64) / self.T


def fit_calibrators(entries) -> dict:
    """Z-score and temperature statistics from the validation splits only."""
    td_all = np.concatenate([e["s_td"] for e in entries if e])
    o_all = np.concatenate([e["s_o"] for e in entries if e])
    return {
        "zscore": {"td": ZScore(td_all.mean(), td_all.std()),
                   "o": ZScore(o_all.mean(), o_all.std())},
        "stats": {
            "td": {"mean": float(td_all.mean()), "std": float(td_all.std()),
                   "min": float(td_all.min()), "max": float(td_all.max())},
            "o": {"mean": float(o_all.mean()), "std": float(o_all.std()),
                  "min": float(o_all.min()), "max": float(o_all.max())},
        },
    }


def fit_temperature(entries, alpha, grid=(0.5, 1.0, 2.0, 4.0, 8.0)) -> float:
    """One parameter per view, chosen to maximise fused Top-1 on validation."""
    best, best_top1 = 1.0, -1.0
    for T in grid:
        cal = {"td": Temperature(T), "o": Temperature(T)}
        fused = [fuse(e, alpha, cal) for e in entries]
        top1 = agg([rank_of(f["s"], f["ids"], f["target_id"]) for f in fused]).get("top1", 0)
        if top1 > best_top1:
            best, best_top1 = T, top1
    return best


def fuse(entry, alpha, calibrator=None):
    s_td, s_o = entry["s_td"], entry["s_o"]
    if calibrator:
        s_td = calibrator["td"].apply(s_td)
        s_o = calibrator["o"].apply(s_o)
    return {"s": alpha * s_td + (1.0 - alpha) * s_o,
            "ids": entry["ids"], "target_id": entry["target_id"]}


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def rule_alpha(phrase, rule):
    flags = token_flags(phrase)
    app, geo = flags["has_appearance"], flags["has_geometry"]
    if rule == "R1":
        a_td, a_o = (0.3, 0.7) if app else (0.7, 0.3)
    elif rule == "R2":
        a_td, a_o = (0.2, 0.8) if app else (0.8, 0.2)
    else:  # R3
        if app and geo:
            a_td, a_o = (0.5, 0.5)
        elif app:
            a_td, a_o = (0.25, 0.75)
        else:
            a_td, a_o = (0.75, 0.25)
    return a_o / (a_td + a_o), flags


def quality_score(rec, weights):
    """Oblique visual quality in [0, 1] from the recorded per-sample features."""
    q = rec["quality"]
    pixel = float(np.clip(q["target_pixel_ratio"], 0.0, 1.0))
    valid = float(np.clip(q["oblique_valid_pixel_ratio"], 0.0, 1.0))
    frame = 1.0 if q["target_in_frame"] else 0.0
    dist = float(np.clip(rec["target_distance"] / 220.0, 0.0, 1.0))
    raw = (weights["pixel"] * pixel + weights["valid"] * valid
           + weights["frame"] * frame - weights["distance"] * dist)
    return float(np.clip(raw, 0.0, 1.0))


def quality_weighted_alpha(rec, weights, base=1.0):
    q = quality_score(rec, weights)
    lang = base
    return lang * q


# --------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------

def oracle_view(entry):
    a = rank_of(entry["s_td"], entry["ids"], entry["target_id"])
    b = rank_of(entry["s_o"], entry["ids"], entry["target_id"])
    if a is None or b is None:
        return None
    if b["rank"] < a["rank"]:
        return b, "oblique"
    return a, "topdown"


def oracle_candidatewise(entry):
    """A stronger, unattainable bound: the best score any view gives a candidate."""
    s = np.maximum(entry["s_td"], entry["s_o"])
    return rank_of(s, entry["ids"], entry["target_id"])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--variant", default="phrase")
    args = ap.parse_args()

    root = Path(args.dir) if args.dir else \
        Path(__file__).resolve().parent.parent / "artifacts" / "fusion"
    scores_dir = root / "scores"
    (root / "figures").mkdir(parents=True, exist_ok=True)

    data = {s: load_split(scores_dir, s)
            for s in ("train_seen", "val_seen", "val_unseen")}
    print({k: len(v) for k, v in data.items()})

    validation = data["train_seen"] + data["val_seen"]
    out = {"variant": args.variant,
           "counts": {k: len(v) for k, v in data.items()}}

    # ---- 1. TD_OPT chosen on validation only -------------------------------
    td_table = {}
    for view in VIEWS_TD:
        entries = scores_for(validation, view, args.variant)
        td_table[view] = agg([rank_of(e["s_td"], e["ids"], e["target_id"])
                              for e in entries if e])
    order = sorted(td_table, key=lambda v: (-td_table[v]["top1"],
                                            -td_table[v]["mrr"],
                                            -td_table[v]["positive_margin_ratio"]))
    td_opt = order[0]
    out["topdown_operating_point"] = {"table": td_table, "chosen": td_opt,
                                      "chosen_on": "train_seen+val_seen"}

    # ---- 2. calibration on validation --------------------------------------
    val_entries = scores_for(validation, td_opt, args.variant)
    val_entries = [e for e in val_entries if e]
    cal = fit_calibrators(val_entries)
    T = fit_temperature(val_entries, 0.5)
    out["score_calibration"] = {
        "raw_stats": cal["stats"],
        "zscore": {k: {"mu": v.mu, "sigma": v.sigma}
                   for k, v in cal["zscore"].items()},
        "temperature": T,
    }

    calib_variants = {
        "C0_raw": None,
        "C1_zscore": cal["zscore"],
        "C2_temperature": {"td": Temperature(T), "o": Temperature(T)},
    }

    # ---- 3. fixed alpha sweep on validation --------------------------------
    sweep = {}
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for name, calibrator in calib_variants.items():
        rows = {}
        for alpha in alphas:
            metrics = [rank_of(*(lambda f: (f["s"], f["ids"], f["target_id"]))(
                fuse(e, alpha, calibrator))) for e in val_entries]
            rows[f"{alpha:.2f}"] = agg(metrics)
        sweep[name] = rows
    best_calib = max(sweep, key=lambda c: max(sweep[c][a]["top1"] for a in sweep[c]))
    best_alpha = max(sweep[best_calib],
                     key=lambda a: (sweep[best_calib][a]["top1"],
                                    sweep[best_calib][a]["mrr"]))
    out["fixed_fusion"] = {"sweep": sweep, "chosen_calibration": best_calib,
                           "chosen_alpha": float(best_alpha)}

    # ---- 4. rule gate on validation ----------------------------------------
    rule_rows = {}
    for rule in ("R1", "R2", "R3"):
        metrics = []
        for e in val_entries:
            a_o, _ = rule_alpha(e["rec"]["texts"].get(args.variant) or "", rule)
            metrics.append(rank_of(*(lambda f: (f["s"], f["ids"], f["target_id"]))(
                fuse(e, 1.0 - a_o))))
        rule_rows[rule] = agg(metrics)
    best_rule = max(rule_rows, key=lambda r: (rule_rows[r]["top1"], rule_rows[r]["mrr"]))
    out["rule_gate"] = {"table": rule_rows, "chosen": best_rule}

    # ---- 5. quality-aware gate on validation --------------------------------
    q_grid = [{"pixel": p, "valid": v, "frame": f, "distance": d}
              for p in (0.5, 1.0) for v in (0.5, 1.0) for f in (0.2, 0.5)
              for d in (0.0, 0.25)]
    q_rows = []
    for w in q_grid:
        metrics = []
        for e in val_entries:
            a_o = quality_weighted_alpha(e["rec"], w)
            metrics.append(rank_of(*(lambda f: (f["s"], f["ids"], f["target_id"]))(
                fuse(e, 1.0 - a_o))))
        q_rows.append({"weights": w, **agg(metrics)})
    best_q = max(q_rows, key=lambda r: (r["top1"], r["mrr"]))
    out["quality_gate"] = {"grid": q_rows, "chosen": best_q["weights"],
                           "validation_top1": best_q["top1"],
                           "chosen_on": "train_seen+val_seen"}

    # ---- 6. learned gate, only if the oracle says there is signal ----------
    oracle_val = []
    for e in val_entries:
        ov = oracle_view(e)
        if ov:
            oracle_val.append(ov[0])
    oracle_val_agg = agg(oracle_val)
    td_val_agg = agg([rank_of(e["s_td"], e["ids"], e["target_id"]) for e in val_entries])
    o_val_agg = agg([rank_of(e["s_o"], e["ids"], e["target_id"]) for e in val_entries])
    out["oracle"] = {
        "validation": {"per_view_oracle": oracle_val_agg, "topdown": td_val_agg,
                       "oblique": o_val_agg},
        "validation_candidatewise": agg(
            [oracle_candidatewise(e) for e in val_entries]),
    }

    learned = None
    headroom = oracle_val_agg["top1"] - max(td_val_agg["top1"], o_val_agg["top1"])
    out["oracle_headroom_over_best_single_view"] = float(headroom)
    if headroom >= 0.05:
        learned = train_learned_gate(data, td_opt, args.variant, cal)
        out["learned_gate"] = learned
    else:
        out["learned_gate"] = {"skipped": True,
                               "reason": f"oracle headroom {headroom:+.3f} < 0.05",
                               "chosen_on": "train_seen+val_seen"}

    # ---- 7. test split, scored once ---------------------------------------
    test_entries = [e for e in scores_for(data["val_unseen"], td_opt, args.variant) if e]
    seen_entries = [e for e in
                    scores_for(data["val_seen"], td_opt, args.variant) if e]
    if not test_entries:
        print("no val_unseen entries; nothing to score")
        return
    out["final"] = final_eval(test_entries, seen_entries, out, learned, cal)

    out["figures"] = write_figures(root, test_entries, learned, out)

    # The fitted gate lives in memory as a module plus a feature builder; only
    # the numbers belong in the report.
    if isinstance(out.get("learned_gate"), dict):
        out["learned_gate"] = {k: v for k, v in out["learned_gate"].items()
                               if k not in ("module", "features")}
    suffix = "" if args.variant == "phrase" else f"_{args.variant}"
    (root / f"metrics{suffix}.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=float) + "\n")
    write_csv(root / f"metrics{suffix}.csv", out)
    write_manifest(root / (f"fusion_eval_manifest_v1{suffix}.json" if suffix
                           else "fusion_eval_manifest_v1.json"),
                   data, td_opt, args.variant)

    print(json.dumps({k: out[k] for k in
                      ("counts", "topdown_operating_point", "fixed_fusion",
                       "rule_gate", "oracle_headroom_over_best_single_view")},
                     indent=2, default=float))
    print("\n--- final (val_unseen, chosen on val_seen/train_seen) ---")
    for name, m in out["final"]["table"].items():
        print(f"  {name:22s} n={m['n']:4d} top1={m['top1']:.3f} top4={m['top4']:.3f} "
              f"mrr={m['mrr']:.3f} marg_med={m['margin_median']:+.4f} "
              f"pos={m['positive_margin_ratio']:.3f}")
    print(f"\nwrote {root / 'metrics.json'}")


def train_learned_gate(data, td_opt, variant, cal):
    """A 2-layer MLP over cheap features, supervised by which view ranks better."""
    import torch
    import torch.nn as nn

    def features(entries):
        rows = []
        for e in entries:
            rec = e["rec"]
            flags = token_flags(rec["texts"].get(variant) or "")
            q = rec["quality"]
            rows.append([
                float(flags["has_appearance"]), float(flags["has_geometry"]),
                float(len(flags["appearance"])),
                float(np.clip(rec["target_distance"] / 220.0, 0, 1)),
                float(np.clip(rec["same_class_candidates"] / 10.0, 0, 1)),
                float(np.clip(q["target_pixel_ratio"], 0, 1)),
                float(np.clip(q["oblique_valid_pixel_ratio"], 0, 1)),
                float(q["target_in_frame"]),
                float(e["s_td"].max() - e["s_td"].min()),
                float(e["s_o"].max() - e["s_o"].min()),
                float(e["s_td"].mean()), float(e["s_o"].mean()),
            ])
        return torch.tensor(rows, dtype=torch.float32)

    def labels(entries):
        y = []
        for e in entries:
            a = rank_of(e["s_td"], e["ids"], e["target_id"])
            b = rank_of(e["s_o"], e["ids"], e["target_id"])
            if a["rank"] < b["rank"]:
                y.append(1.0)      # top-down is the better view -> high alpha
            elif b["rank"] < a["rank"]:
                y.append(0.0)
            else:
                y.append(0.5)
        return torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    train = [e for e in scores_for(data["train_seen"], td_opt, variant) if e]
    val = [e for e in scores_for(data["val_seen"], td_opt, variant) if e]
    if len(train) < 20 or len(val) < 10:
        return {"skipped": True, "reason": "not enough training samples"}

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1),
                        nn.Sigmoid())
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCELoss()
    X, y = features(train), labels(train)
    Xv, yv = features(val), labels(val)
    best_state, best_top1 = None, -1.0
    for epoch in range(300):
        net.train()
        opt.zero_grad()
        loss = lossf(net(X), y)
        loss.backward()
        opt.step()
        if epoch % 25 == 0:
            net.eval()
            with torch.no_grad():
                alpha = net(Xv).squeeze(1).numpy()
            metrics = [rank_of(*(lambda f: (f["s"], f["ids"], f["target_id"]))(
                fuse(e, float(a)))) for e, a in zip(val, alpha)]
            top1 = agg(metrics).get("top1", 0.0)
            if top1 > best_top1:
                best_top1 = top1
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    torch.save({"state_dict": net.state_dict(), "n_params": int(n_params)},
               str(Path(__file__).resolve().parent.parent / "artifacts" / "fusion"
                   / "learned_gate.pt"))
    return {"n_params": int(n_params), "validation_top1": float(best_top1),
            "chosen_on": "train_seen (fit) / val_seen (early stop)", "module": net,
            "features": features}


def final_eval(test_entries, seen_entries, out, learned, cal):
    """Score every method once on val_unseen, with all choices already frozen."""
    td_opt = out["topdown_operating_point"]["chosen"]
    alpha = out["fixed_fusion"]["chosen_alpha"]
    calib_name = out["fixed_fusion"]["chosen_calibration"]
    calibrator = {"C0_raw": None, "C1_zscore": cal["zscore"],
                  "C2_temperature": {"td": Temperature(out["score_calibration"]["temperature"]),
                                     "o": Temperature(out["score_calibration"]["temperature"])}}[calib_name]
    rule = out["rule_gate"]["chosen"]
    qw = out["quality_gate"]["chosen"]

    def m(fused):
        return rank_of(fused["s"], fused["ids"], fused["target_id"])

    methods = {
        "TD_OPT": [rank_of(e["s_td"], e["ids"], e["target_id"]) for e in test_entries],
        "O2048": [rank_of(e["s_o"], e["ids"], e["target_id"]) for e in test_entries],
        "Fixed_0.5": [m(fuse(e, 0.5, calibrator)) for e in test_entries],
        f"Fixed_best_{alpha}": [m(fuse(e, alpha, calibrator)) for e in test_entries],
        "Rule_gate": [m(fuse(e, 1.0 - rule_alpha(
            e["rec"]["texts"].get(out["variant"]) or "", rule)[0])) for e in test_entries],
        "Rule_plus_quality": [m(fuse(e, 1.0 - quality_weighted_alpha(
            e["rec"], qw))) for e in test_entries],
        "Oracle_per_view": [oracle_view(e)[0] if oracle_view(e) else None
                            for e in test_entries],
        "Oracle_candidatewise": [oracle_candidatewise(e) for e in test_entries],
    }
    if learned and not learned.get("skipped"):
        net = learned["module"]
        import torch
        with torch.no_grad():
            a = net(learned["features"](test_entries)).squeeze(1).numpy()
        methods["Learned_gate"] = [m(fuse(e, float(x))) for e, x in zip(test_entries, a)]
        methods["Learned_gate_calibrated"] = [
            m(fuse(e, float(x), calibrator)) for e, x in zip(test_entries, a)]

    table = {k: agg(v) for k, v in methods.items()}

    best_single = max(table["TD_OPT"]["top1"], table["O2048"]["top1"])
    non_oracle = {k: v for k, v in table.items() if not k.startswith("Oracle")}
    best_fusion = max((k for k in non_oracle if k not in ("TD_OPT", "O2048")),
                      key=lambda k: (table[k]["top1"], table[k]["mrr"]), default=None)

    transitions = {}
    for name in ("Fixed_0.5", f"Fixed_best_{alpha}", "Rule_gate", "Rule_plus_quality",
                 "Learned_gate"):
        if name not in methods:
            continue
        transitions[f"TD_vs_{name}"] = paired(methods["TD_OPT"], methods[name])
        transitions[f"O_vs_{name}"] = paired(methods["O2048"], methods[name])
        transitions[f"both_single_fail_{name}"] = int(sum(
            1 for a, b, c in zip(methods["TD_OPT"], methods["O2048"], methods[name])
            if a and b and c and not a["top1"] and not b["top1"] and c["top1"]))

    # view-choice accuracy of the gates
    choice = {}
    for name in ("Rule_gate", "Rule_plus_quality", "Learned_gate"):
        if name not in methods:
            continue
        correct, total = 0, 0
        for e in test_entries:
            a = rank_of(e["s_td"], e["ids"], e["target_id"])
            b = rank_of(e["s_o"], e["ids"], e["target_id"])
            if a["rank"] == b["rank"]:
                continue
            if name == "Rule_gate":
                a_o = rule_alpha(e["rec"]["texts"].get(out["variant"]) or "", rule)[0]
            elif name == "Rule_plus_quality":
                a_o = quality_weighted_alpha(e["rec"], qw)
            else:
                import torch
                with torch.no_grad():
                    a_o = 1.0 - float(learned["module"](
                        learned["features"]([e])).squeeze())
            chose_td = a_o < 0.5
            better_td = a["rank"] < b["rank"]
            correct += int(chose_td == better_td)
            total += 1
        choice[name] = {"accuracy": correct / max(total, 1), "n": total}

    # ---- buckets: attribute type, same-class ambiguity, distance -----------
    def bucket_report(key_fn):
        groups = {}
        for i, e in enumerate(test_entries):
            groups.setdefault(key_fn(e), []).append(i)
        out_b = {}
        for name, idxs in sorted(groups.items()):
            out_b[name] = {"n": len(idxs)}
            for method, metrics in methods.items():
                sub = [metrics[i] for i in idxs]
                a = agg(sub)
                if a:
                    out_b[name][method] = {
                        "top1": a["top1"], "mrr": a["mrr"],
                        "margin_median": a["margin_median"],
                        "positive_margin_ratio": a["positive_margin_ratio"]}
        return out_b

    def phrase_kind(e):
        f = token_flags(e["rec"]["texts"].get("phrase") or "")
        if f["has_appearance"] and f["has_geometry"]:
            return "appearance+geometry"
        if f["has_appearance"]:
            return "appearance_only"
        if f["has_geometry"]:
            return "geometry_only"
        return "category_only"

    def attribute_type(e):
        f = token_flags(e["rec"]["texts"].get("phrase") or "")
        kinds = sorted(f["appearance"].keys())
        return kinds[0] if kinds else ("geometry" if f["has_geometry"] else "none")

    buckets = {
        "phrase_kind": bucket_report(phrase_kind),
        "attribute_type": bucket_report(attribute_type),
        "same_class": bucket_report(
            lambda e: "0" if e["rec"]["same_class_candidates"] == 0
            else ("1-3" if e["rec"]["same_class_candidates"] <= 3 else ">=4")),
        "distance": bucket_report(
            lambda e: "near<50m" if e["rec"]["target_distance"] < 50
            else ("medium50-100m" if e["rec"]["target_distance"] < 100 else "far>100m")),
    }

    return {
        "n": len(test_entries),
        "table": table,
        "buckets": buckets,
        "best_single_view_top1": best_single,
        "best_fusion": best_fusion,
        "best_fusion_delta": (table[best_fusion]["top1"] - best_single) if best_fusion else None,
        "transitions": transitions,
        "view_choice_accuracy": choice,
        "frozen_choices": {"td_operating_point": td_opt,
                           "calibration": calib_name, "alpha": alpha,
                           "rule": rule, "quality_weights": qw},
    }


def write_figures(root: Path, test_entries, learned, out) -> dict:
    """Score distributions, the alpha the gates choose, and how they relate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    def save(fig, name):
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written[name] = str(path)

    # score distributions: positives vs hardest negatives, per view
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, label in ((axes[0], "s_td", "top-down"), (axes[1], "s_o", "oblique45")):
        pos, hard = [], []
        for e in test_entries:
            i = e["ids"].index(e["target_id"])
            s = np.asarray(e[key])
            others = np.delete(s, i)
            pos.append(s[i])
            hard.append(others.max())
        ax.hist(pos, bins=30, alpha=0.65, label="target", color="#2b6cb0")
        ax.hist(hard, bins=30, alpha=0.65, label="hardest negative", color="#c05621")
        ax.set_title(f"{label}  (n={len(pos)})")
        ax.set_xlabel("cosine similarity")
        ax.legend(fontsize=8)
    save(fig, "score_distributions.png")

    # top-down vs oblique score scatter for the target
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    xs = [e["s_td"][e["ids"].index(e["target_id"])] for e in test_entries]
    ys = [e["s_o"][e["ids"].index(e["target_id"])] for e in test_entries]
    ax.scatter(xs, ys, s=9, alpha=0.6)
    lim = [min(min(xs), min(ys)) - 0.01, max(max(xs), max(ys)) + 0.01]
    ax.plot(lim, lim, "--", lw=0.8, color="gray")
    ax.set_xlabel("top-down target similarity")
    ax.set_ylabel("oblique45 target similarity")
    ax.set_title("per-sample target scores")
    save(fig, "score_scatter.png")

    # alpha the gates choose
    if learned and not learned.get("skipped"):
        import torch
        with torch.no_grad():
            alphas = learned["module"](learned["features"](test_entries)).squeeze(1).numpy()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        axes[0].hist(alphas, bins=25, color="#553c9a")
        axes[0].set_title("learned gate alpha (1 = trust top-down)")
        axes[0].set_xlabel("alpha")
        rich = [a for a, e in zip(alphas, test_entries)
                if token_flags(e["rec"]["texts"].get(out["variant"]) or "")["has_appearance"]]
        poor = [a for a, e in zip(alphas, test_entries)
                if not token_flags(e["rec"]["texts"].get(out["variant"]) or "")["has_appearance"]]
        axes[1].hist([rich, poor], bins=20, stacked=True,
                     label=[f"appearance-rich (n={len(rich)})",
                            f"category/none (n={len(poor)})"],
                     color=["#2f855a", "#b7791f"])
        axes[1].legend(fontsize=8)
        axes[1].set_title("alpha by phrase type")
        save(fig, "alpha_distribution.png")

        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        dist = [e["rec"]["target_distance"] for e in test_entries]
        ax.scatter(dist, alphas, s=9, alpha=0.6)
        ax.set_xlabel("landmark distance (m)")
        ax.set_ylabel("alpha")
        save(fig, "alpha_vs_distance.png")

    return written


def write_csv(path: Path, out: dict):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n", "top1", "top4", "mrr", "margin_median",
                    "positive_margin_ratio"])
        for name, m in out.get("final", {}).get("table", {}).items():
            w.writerow([name, m.get("n"), round(m.get("top1", 0), 4),
                        round(m.get("top4", 0), 4), round(m.get("mrr", 0), 4),
                        round(m.get("margin_median", 0), 5),
                        round(m.get("positive_margin_ratio", 0), 4)])


def write_manifest(path: Path, data: dict, td_opt: str, variant: str):
    entries = {}
    for split, records in data.items():
        rows = []
        for rec in records:
            cv = common_view(rec, td_opt, VIEW_O, variant)
            if cv is None:
                continue
            rows.append({
                "map": rec["map"], "episode_index": rec["episode_index"],
                "step": rec["step"], "instruction": rec["instruction"],
                "referenced_phrase": rec["texts"].get("phrase"),
                "name": rec["texts"].get("name"),
                "candidate_ids": cv["ids"], "gt_candidate_id": rec["target_id"],
                "same_class_candidates": rec["same_class_candidates"],
                "distance": rec["target_distance"],
                "topdown_source": td_opt, "oblique_source": VIEW_O,
                "candidate_ordering": "as listed in candidate_ids",
            })
        entries[split] = rows
    path.write_text(json.dumps(
        {"version": 1, "variant": variant,
         "note": "Every method in this round is scored on exactly these samples "
                 "and exactly these candidate orderings.",
         "counts": {k: len(v) for k, v in entries.items()}, "splits": entries},
        indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
