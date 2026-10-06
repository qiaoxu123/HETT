#!/usr/bin/env python3
"""The tables, the transition counts and the verdict for the entity-grounding round.

Reads what the trainer wrote and answers the questions the round was asked, with
one rule kept throughout: **which method is best is decided on val_seen**, and
val_unseen is then read once for that decision.  The full val_unseen table is
also printed, because a reader is entitled to see every method's held-out number
-- but the pass/fail call uses the pair chosen before val_unseen was touched.

Everything is reported per entity group as well as overall.  A gain that lives
entirely in buildings, with vehicles flat or worse, is not a general
target-entity result, and the per-group columns are what make that visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sensaturban_fpv.config import artifact_dir, load_config  # noqa: E402

GROUPS = ("building", "vehicle", "other")
VIEW_OF = {"TD_global": "TD", "TD_masked": "TD", "TD_tight": "TD",
           "O_global": "O", "O_masked": "O", "O_tight": "O"}


def load_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_method = defaultdict(list)
    for row in rows:
        by_method[(row["method"], row["split"])].append(row)
    return by_method


def seed_average(rows):
    """One record per (method, split, sample), averaged over the seed repeats."""
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["key"],)].append(row)
    out = {}
    for (key,), entries in by_key.items():
        out[key] = {
            "key": key, "map": entries[0]["map"], "group": entries[0]["group"],
            "target_type": entries[0]["target_type"],
            "target_distance": entries[0]["target_distance"],
            "rank": float(np.mean([e["rank"] for e in entries])),
            "margin": float(np.mean([e["margin"] for e in entries])),
            "top1": float(np.mean([bool(e["top1"]) for e in entries])),
            "top1_vote": bool(np.mean([bool(e["top1"]) for e in entries]) >= 0.5),
            "top4": float(np.mean([bool(e["top4"]) for e in entries])),
            "mrr": float(np.mean([e["mrr"] for e in entries])),
            "positive_margin": float(np.mean(
                [bool(e["positive_margin"]) for e in entries])),
        }
    return out


def aggregate(entries):
    if not entries:
        return {"n": 0}
    values = list(entries.values()) if isinstance(entries, dict) else list(entries)
    top1 = np.array([e["top1"] for e in values], dtype=float)
    return {
        "n": len(values),
        "top1": float(top1.mean()),
        "top4": float(np.mean([e["top4"] for e in values])),
        "mrr": float(np.mean([e["mrr"] for e in values])),
        "median_margin": float(np.median([e["margin"] for e in values])),
        "positive_margin_ratio": float(np.mean(
            [e["positive_margin"] for e in values])),
        "top1_ci": wilson(float(top1.sum()), len(values)),
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(centre - half, 0)), float(min(centre + half, 1))]


def mcnemar(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return {"a_only": n10, "b_only": n01, "n_discordant": 0, "p_value": 1.0}
    k = min(n10, n01)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"a_only": n10, "b_only": n01, "n_discordant": n,
            "p_value": float(min(2 * tail, 1.0))}


def paired(a_entries, b_entries):
    keys = sorted(set(a_entries) & set(b_entries))
    a = np.array([a_entries[k]["top1_vote"] for k in keys], dtype=bool)
    b = np.array([b_entries[k]["top1_vote"] for k in keys], dtype=bool)
    stats = mcnemar(a, b)
    stats.update({
        "n": len(keys),
        "both_correct": int(np.sum(a & b)),
        "both_wrong": int(np.sum(~a & ~b)),
        "a_only_correct": int(np.sum(a & ~b)),
        "b_only_correct": int(np.sum(~a & b)),
    })
    return stats


def transitions(td_entries, o_entries, fusion_entries, group=None):
    """The four categories the round asks for, on the samples all three scored."""
    keys = sorted(set(td_entries) & set(o_entries) & set(fusion_entries))
    if group:
        keys = [k for k in keys if fusion_entries[k]["group"] == group]
    out = {"n": len(keys), "td_wrong_o_wrong_fusion_correct": 0,
           "td_correct_o_wrong_fusion_correct": 0,
           "td_wrong_o_correct_fusion_correct": 0,
           "either_correct_fusion_wrong": 0,
           "td_correct_o_correct_fusion_correct": 0,
           "all_wrong": 0}
    for key in keys:
        td = td_entries[key]["top1_vote"]
        ob = o_entries[key]["top1_vote"]
        fu = fusion_entries[key]["top1_vote"]
        if td and ob and fu:
            out["td_correct_o_correct_fusion_correct"] += 1
        elif not td and not ob and fu:
            out["td_wrong_o_wrong_fusion_correct"] += 1
        elif td and not ob and fu:
            out["td_correct_o_wrong_fusion_correct"] += 1
        elif not td and ob and fu:
            out["td_wrong_o_correct_fusion_correct"] += 1
        elif (td or ob) and not fu:
            out["either_correct_fusion_wrong"] += 1
        else:
            out["all_wrong"] += 1
    return out


def write_figures(report, out_dir: Path, variant: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unseen = report["val_unseen"]
    names = list(unseen.keys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    y = np.arange(len(names))
    vals = [unseen[n]["overall"]["top1"] for n in names]
    lo = [unseen[n]["overall"]["top1"] - unseen[n]["overall"]["top1_ci"][0] for n in names]
    hi = [unseen[n]["overall"]["top1_ci"][1] - unseen[n]["overall"]["top1"] for n in names]
    colours = ["#4c72b0" if n in report["frozen"] else
               ("#c44e52" if "Shuffle" in n else "#55a868") for n in names]
    ax.barh(y, vals, xerr=[lo, hi], color=colours, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Top-1 on val_unseen")
    ax.set_title(f"val_unseen Top-1 ({variant}), 95% Wilson")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    width = 0.27
    for j, group in enumerate(GROUPS):
        vals = [unseen[n][group]["top1"] if unseen[n][group].get("n") else np.nan
                for n in names]
        ax.barh(y + (j - 1) * width, vals, height=width, label=group)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Top-1 on val_unseen")
    ax.set_title("by entity group")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"entity_grounding_val_unseen_{variant}.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    keys = list(report["transitions"].keys())
    data = np.array([[report["transitions"][k][c] for c in
                      ("td_wrong_o_wrong_fusion_correct",
                       "td_correct_o_wrong_fusion_correct",
                       "td_wrong_o_correct_fusion_correct",
                       "either_correct_fusion_wrong")] for k in keys])
    bottom = np.zeros(len(keys))
    labels = ["TD wrong + O wrong -> fusion correct",
              "TD correct + O wrong -> fusion correct",
              "TD wrong + O correct -> fusion correct",
              "either single correct -> fusion wrong"]
    for j, label in enumerate(labels):
        ax.bar(keys, data[:, j], bottom=bottom, label=label)
        bottom += data[:, j]
    ax.set_ylabel("samples")
    ax.set_title(f"transitions on val_unseen ({variant})")
    ax.legend(fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"entity_grounding_transitions_{variant}.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, group in zip(axes, GROUPS):
        for name, colour in ((report["reference_fusion"], "#55a868"),
                             (report["reference_single"], "#4c72b0")):
            entries = report["entries"][name]["val_unseen"]
            values = [e["margin"] for k, e in sorted(entries.items())
                      if e["group"] == group]
            if values:
                ax.hist(values, bins=24, alpha=0.55, label=name, color=colour)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{group}")
        ax.set_xlabel("margin (target - best other)")
        ax.legend(fontsize=7)
    fig.suptitle(f"paired margins on val_unseen ({variant})")
    fig.tight_layout()
    fig.savefig(out_dir / f"entity_grounding_margins_{variant}.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--variant", default="phrase")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(args.results) if args.results else \
        artifact_dir(cfg) / "entity_grounding" / "results"
    metrics_path = results_dir / f"metrics_{args.variant}.json"
    metrics = json.loads(metrics_path.read_text())
    by_method = load_rows(results_dir / f"persample_{args.variant}.jsonl")

    entries = {}
    for (method, split), rows in by_method.items():
        entries.setdefault(method, {})[split] = seed_average(rows)

    frozen = [n for n in metrics["methods"] if metrics["methods"][n]["kind"] == "frozen"]
    learned = [n for n in metrics["methods"] if metrics["methods"][n]["kind"] == "learned"]

    def summarise(split):
        out = {}
        for name in frozen + learned:
            if split not in entries.get(name, {}):
                continue
            overall = aggregate(entries[name][split])
            out[name] = {"overall": overall, "params": metrics["methods"][name]["trainable_params"],
                         "fusion_level": metrics["methods"][name]["fusion_level"]}
            for group in GROUPS:
                out[name][group] = aggregate(
                    {k: v for k, v in entries[name][split].items()
                     if v["group"] == group})
        return out

    val_seen, val_unseen = summarise("val_seen"), summarise("val_unseen")

    # Selection happens on val_seen, never on the held-out split.
    single_names = [n for n in frozen]
    best_single = max(single_names, key=lambda n: (val_seen[n]["overall"]["top1"],
                                                   val_seen[n]["overall"]["mrr"]))
    best_fusion = max(learned, key=lambda n: (val_seen[n]["overall"]["top1"],
                                              val_seen[n]["overall"]["mrr"]))
    td_ref = max([n for n in frozen if VIEW_OF.get(n) == "TD"],
                 key=lambda n: val_seen[n]["overall"]["top1"])
    o_ref = max([n for n in frozen if VIEW_OF.get(n) == "O"],
                key=lambda n: val_seen[n]["overall"]["top1"])

    report = {"variant": args.variant, "frozen": frozen, "learned": learned,
              "counts": metrics["counts"], "target_groups": metrics["target_groups"],
              "target_types": metrics["target_types"],
              "candidate_counts": metrics["candidate_counts"],
              "support_sources": metrics["support_sources"],
              "correspondence": metrics["correspondence"],
              "target_distance_median": metrics["target_distance_median"],
              "tuning": metrics["tuning"],
              "val_seen": val_seen, "val_unseen": val_unseen,
              "reference_single": best_single, "reference_fusion": best_fusion,
              "td_reference": td_ref, "o_reference": o_ref,
              "entries": entries}

    # held-out comparison of the pair chosen on val_seen
    report["paired"] = {}
    for split in ("val_seen", "val_unseen"):
        report["paired"][split] = {
            "fusion_vs_best_single": paired(entries[best_fusion][split],
                                            entries[best_single][split]),
            "fusion_vs_td": paired(entries[best_fusion][split],
                                   entries[td_ref][split]),
            "fusion_vs_o": paired(entries[best_fusion][split],
                                  entries[o_ref][split]),
        }
    report["transitions"] = {
        f"{best_fusion}_vs_TD+O": transitions(entries[td_ref]["val_unseen"],
                                              entries[o_ref]["val_unseen"],
                                              entries[best_fusion]["val_unseen"]),
        f"{best_fusion}_vs_{best_single}": transitions(
            entries[best_single]["val_unseen"], entries[best_single]["val_unseen"],
            entries[best_fusion]["val_unseen"]),
    }
    for name in learned:
        if "Shuffle" in name:
            report["transitions"][f"{name}_vs_TD+O"] = transitions(
                entries[td_ref]["val_unseen"], entries[o_ref]["val_unseen"],
                entries[name]["val_unseen"])

    report["per_group_paired"] = {}
    for group in GROUPS:
        a = {k: v for k, v in entries[best_fusion]["val_unseen"].items()
             if v["group"] == group}
        b = {k: v for k, v in entries[best_single]["val_unseen"].items()
             if v["group"] == group}
        report["per_group_paired"][group] = paired(a, b)

    # --- ablation: real correspondence against the shuffled control
    shuffle_name = next((n for n in learned if "Shuffle" in n), None)
    aligned = next((n for n in learned if n.startswith("GeoAligned")
                    and "XL" not in n), None)
    report["alignment_control"] = (
        paired(entries[aligned]["val_unseen"], entries[shuffle_name]["val_unseen"])
        if shuffle_name and aligned else None)

    # --- verdict
    fusion_top1 = val_unseen[best_fusion]["overall"]["top1"]
    single_top1 = val_unseen[best_single]["overall"]["top1"]
    gain = fusion_top1 - single_top1
    secondary_ok = (
        val_unseen[best_fusion]["overall"]["mrr"] > val_unseen[best_single]["overall"]["mrr"]
        and val_unseen[best_fusion]["overall"]["median_margin"]
        > val_unseen[best_single]["overall"]["median_margin"]
        and val_unseen[best_fusion]["overall"]["positive_margin_ratio"]
        > val_unseen[best_single]["overall"]["positive_margin_ratio"])
    alignment_gain = None
    if report["alignment_control"] is not None:
        alignment_gain = (val_unseen[aligned]["overall"]["top1"]
                          - val_unseen[shuffle_name]["overall"]["top1"])
    vehicle_delta = (val_unseen[best_fusion]["vehicle"]["top1"]
                     - val_unseen[best_single]["vehicle"]["top1"]) \
        if val_unseen[best_fusion]["vehicle"].get("n") else None

    if gain >= 0.05 and secondary_ok and (alignment_gain or 0) > 0.02 \
            and (vehicle_delta is None or vehicle_delta > -0.02):
        verdict = "STRONG PASS"
    elif gain >= 0.02 and secondary_ok:
        verdict = "WEAK PASS"
    else:
        verdict = "FAIL"
    report["verdict"] = {
        "call": verdict, "gain_top1": gain, "secondary_metrics_improved": secondary_ok,
        "alignment_gain_over_shuffle": alignment_gain,
        "vehicle_top1_delta": vehicle_delta,
        "fusion": best_fusion, "single": best_single,
        "note": "selection on val_seen; val_unseen read once for this pair",
    }

    out_dir = Path(args.out) if args.out else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # The per-sample entries stay in memory for the figures; writing a quarter of
    # a million of them into the summary would bury the numbers a reader wants.
    serialisable = {k: v for k, v in report.items() if k != "entries"}
    (out_dir / f"analysis_{args.variant}.json").write_text(
        json.dumps(serialisable, indent=2, sort_keys=True, default=float) + "\n")
    write_figures(report, out_dir, args.variant)

    print(f"\n=== main table, chosen on val_seen: single={best_single} "
          f"fusion={best_fusion} ===")
    header = f"{'method':16s} {'params':>8s} {'top1':>6s} {'top4':>6s} {'mrr':>6s} " \
             f"{'med.marg':>9s} {'pos.marg':>9s}"
    for split, table in (("val_seen", val_seen), ("val_unseen", val_unseen)):
        print(f"\n--- {split} ---")
        print(header)
        for name in frozen + learned:
            if name not in table:
                continue
            m = table[name]["overall"]
            print(f"{name:16s} {table[name]['params']:8d} {m['top1']:6.3f} "
                  f"{m['top4']:6.3f} {m['mrr']:6.3f} {m['median_margin']:+9.4f} "
                  f"{m['positive_margin_ratio']:9.3f}")

    print("\n--- per group, val_unseen ---")
    print(f"{'method':16s} " + " ".join(f"{g:>10s}" for g in GROUPS))
    for name in frozen + learned:
        if name not in val_unseen:
            continue
        cells = " ".join(
            f"{val_unseen[name][g]['top1']:10.3f}" if val_unseen[name][g].get("n")
            else f"{'-':>10s}" for g in GROUPS)
        print(f"{name:16s} {cells}")

    print("\n--- transitions, val_unseen ---")
    for key, value in report["transitions"].items():
        print(f"  {key}")
        for name, count in value.items():
            print(f"      {name:38s} {count}")
    print("\n--- paired (McNemar, majority vote over seeds) ---")
    for split, items in report["paired"].items():
        for key, value in items.items():
            print(f"  {split:11s} {key:24s} n={value['n']} "
                  f"b_only={value['b_only_correct']} a_only={value['a_only_correct']} "
                  f"p={value['p_value']:.4f}")
    print("\n--- alignment control ---")
    print(f"  {report['alignment_control']}")
    print(f"\nverdict: {verdict}  gain={gain:+.4f}  secondary_ok={secondary_ok}  "
          f"alignment_gain={alignment_gain}  vehicle_delta={vehicle_delta}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
