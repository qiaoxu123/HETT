#!/usr/bin/env python3
"""Top-K belief, visual rerank, and oracle bounds from an offline candidate export."""
import argparse
import json
from pathlib import Path

import numpy as np

from multiagent.visual_goal.candidate_rerank import oracle_order, rerank_topk


def _recall_at_order(order_xy, target_xy, k, radius=20.0):
    xy = np.asarray(order_xy, dtype=float)[:k]
    d = np.linalg.norm(xy - np.asarray(target_xy, dtype=float)[None, :], axis=1)
    return bool(np.any(d <= radius))


def _top1_distance(order_xy, target_xy):
    return float(np.linalg.norm(np.asarray(order_xy[0], dtype=float) - np.asarray(target_xy, dtype=float)))


def _evaluate_rows(rows, alpha):
    k_values = (1, 4, 8, 16)
    methods = ("belief", "oracle_a_full_rgb", "oracle_b_target_anchor",
               "oracle_c_minimal_template", "oracle_d_geometry")
    out = {m: {f"R@{k}/20": [] for k in k_values} | {"top1_distance_m": []} for m in methods}
    for row in rows:
        xy = np.asarray(row["candidate_xy"], dtype=float)
        target = np.asarray(row["target_xy"], dtype=float)
        belief_order = np.argsort(-np.asarray(row["belief_scores"], dtype=float), kind="stable")
        full_order = np.argsort(-np.asarray(row["full_rgb_scores"], dtype=float), kind="stable")
        ta_order = np.argsort(-np.asarray(row["target_anchor_scores"], dtype=float), kind="stable")
        minimal_order = np.argsort(-np.asarray(row["minimal_template_scores"], dtype=float), kind="stable")
        geom_order = oracle_order(xy, target)
        orders = {
            "belief": belief_order,
            "oracle_a_full_rgb": full_order,
            "oracle_b_target_anchor": ta_order,
            "oracle_c_minimal_template": minimal_order,
            "oracle_d_geometry": geom_order,
        }
        for name in methods:
            ordered_xy = xy[orders[name]]
            for k in k_values:
                out[name][f"R@{k}/20"].append(float(_recall_at_order(ordered_xy, target, k)))
            out[name]["top1_distance_m"].append(_top1_distance(ordered_xy, target))
        for k in (4, 8, 16):
            top = belief_order[:k]
            for key, score_field in (("visual_rerank_full_rgb", "full_rgb_scores"),
                                     ("visual_rerank_target_anchor", "target_anchor_scores"),
                                     ("visual_rerank_minimal_template", "minimal_template_scores")):
                if key not in out:
                    out[key] = {f"R@{kk}/20": [] for kk in k_values} | {"top1_distance_m": []}
                fused = rerank_topk(np.asarray(row["belief_scores"])[top],
                                    np.asarray(row[score_field])[top], xy[top], target, alpha)
                ordered_xy = xy[top][fused["order"]]
                for kk in k_values:
                    out[key][f"R@{kk}/20"].append(float(_recall_at_order(ordered_xy, target, kk)))
                out[key]["top1_distance_m"].append(_top1_distance(ordered_xy, target))
    return {method: {metric: float(np.mean(values)) if values else None for metric, values in metrics.items()}
            for method, metrics in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--gate-json", type=Path, required=True,
                   help="full_goal_retrieval.json; this stage is blocked unless Gate 1 passed")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    gate = json.loads(args.gate_json.read_text()).get("gate1", {})
    if not gate.get("pass"):
        raise SystemExit("Full RGB Gate 1 failed; stop before candidate rerank/oracle")
    rows = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    if any(r.get("split") == "test_unseen" for r in rows):
        raise ValueError("test_unseen may not be accessed")
    # Select lambda on val_seen only, then freeze it for val_unseen reporting.
    seen = [r for r in rows if r.get("split") == "val_seen"]
    unseen = [r for r in rows if r.get("split") == "val_unseen"]
    grid = np.linspace(0, 1, 11)
    calibration = {str(round(float(a), 2)): _evaluate_rows(seen, float(a)) for a in grid}
    alpha = max(grid, key=lambda a: calibration[str(round(float(a), 2))]["visual_rerank_minimal_template"]["R@1/20"])
    results = {"n_val_seen": len(seen), "n_val_unseen": len(unseen),
               "selected_alpha_on_val_seen": float(alpha), "alpha_selection_metric": "visual_rerank_minimal_template R@1/20",
               "val_seen": _evaluate_rows(seen, float(alpha)),
               "val_unseen_heldout": _evaluate_rows(unseen, float(alpha)),
               "candidate_export_schema": "belief + visual scores + candidate XY + offline GT XY labels"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

