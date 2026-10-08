#!/usr/bin/env python3
"""Create a human-readable report and plots from validation-only artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, default=Path("../artifacts/visual_goal_abstraction/dataset_v2"))
    p.add_argument("--eval-json", type=Path, default=Path("../artifacts/visual_goal_abstraction/eval/full_goal_retrieval.json"))
    p.add_argument("--output", type=Path, default=Path("../VISUAL_GOAL_ABSTRACTION_REPORT.md"))
    args = p.parse_args()
    data = json.loads(args.eval_json.read_text())
    stats = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        path = args.dataset_dir / f"dataset_{split}_stats.json"
        if path.exists():
            stats[split] = json.loads(path.read_text())
    result_dir = args.eval_json.parent
    models = data.get("models", {})
    selected = data.get("selected_on_val_seen_only", {})
    gate = data.get("gate1", {})
    gate_seen = gate.get("val_seen", gate)
    gate_unseen = gate.get("val_unseen_heldout", {})

    lines = [
        "# Visual Goal Abstraction / Minimal Sufficient Visual Template",
        "",
        f"Branch: `2027-CVPR/visual-goal-abstraction` (base `a9d95e3`).",
        "",
        "## Protocol and input audit",
        "",
        "Goal templates are 40/80/120 m georeferenced orthographic RGB crops centered on the annotated target position. That GT coordinate is used only by this offline dataset builder. CityNav does not provide captured per-step RGB frames, so query RGB is reconstructed from the same georeferenced orthophoto using an actual pose in the human trajectory and HETT’s orthographic crop geometry; no query is target-centered. Results therefore measure matching within HETT’s orthographic observation domain, not real camera imagery. The vision encoder sees pixel tensors only; labels, map names, coordinates, and candidate identities are used after feature extraction for retrieval metrics.",
        "",
        "The available HETT renderer uses orthographic raster crops. It does not have calibrated oblique/FPV camera geometry, so no perspective warp is presented as a real cross-view observation. This means true top-down-to-oblique/FPV robustness is unavailable in this dataset interface.",
        "",
        "| Split | Annotation rows | unique target scenes | Goal templates | trajectory-pose orthophoto queries | invalid/ground-level poses skipped |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train_seen", "val_seen", "val_unseen"):
        s = stats.get(split)
        if s:
            lines.append(f"| {split} | {s['annotation_rows']} | {s['unique_scenes']} | {s['goal_template_rows']} | {s['trajectory_query_rows']} | {s.get('invalid_or_ground_level_trajectory_poses', 0)} |")
    lines += [
        "",
        "Partial SigLIP2 tuning updated the final two vision blocks for one epoch (3,000 train_seen scene pairs, batch 8, learning rate 1e-5); all 449 val_seen target scenes were excluded from tuning. `val_unseen` is used only for final reporting and example visualization; model/extent selection uses val_seen only. `test_unseen` was not opened. The local tuned checkpoint is `artifacts/visual_goal_abstraction/models/siglip2_partial_trainseen.pt` and is intentionally not committed.",
        "",
        "## A. Full RGB Gate",
        "",
        f"Encoder selected using `val_seen` only: `{selected.get('encoder', 'not selected')}`; selected template extent: {selected.get('extent_m', '—')} m.",
        f"Gate 1: **{'PASS' if gate.get('pass') else 'FAIL'}** — val_seen same-map accuracy {fmt(gate_seen.get('same_map_accuracy'))}, margin {fmt(gate_seen.get('same_map_margin'))}, sign-test p={fmt(gate_seen.get('paired_sign_test_p'), 4)}, distance ρ={fmt(gate_seen.get('distance_spearman_rho'))}; untouched val_unseen same-map accuracy {fmt(gate_unseen.get('same_map_accuracy'))}, margin {fmt(gate_unseen.get('same_map_margin'))}.",
        "",
        "| Encoder | Selected extent (val_seen) | Split | R@1 | R@5 | R@10 | Same-map hard-negative accuracy | AUROC | positive-negative margin | distance Spearman ρ |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, item in models.items():
        ext = str(item.get("val_seen_selection", {}).get("extent_m", "80"))
        for split in ("val_seen", "val_unseen"):
            metric = item.get(split, {}).get("by_extent", {}).get(ext)
            if not metric:
                continue
            hard = metric.get("negative_pools", {}).get("same_map", {})
            lines.append(f"| {model_name} | {ext}m | {split} | {fmt(metric.get('R@1'))} | {fmt(metric.get('R@5'))} | {fmt(metric.get('R@10'))} | {fmt(hard.get('accuracy'))} | {fmt(metric.get('same_map_auroc'))} | {fmt(hard.get('margin'))} | {fmt(metric.get('distance_spearman_rho'))} |")
    selected_name = selected.get("encoder")
    selected_extent = str(selected.get("extent_m", "80"))
    selected_metrics = models.get(selected_name, {})
    lines += ["", f"Selected encoder negative-pool breakdown (`{selected_name}`, {selected_extent} m; selection on val_seen only):", "",
              "| Split | Pool | n | Accuracy | AUROC | Margin |", "|---|---|---:|---:|---:|---:|"]
    for split in ("val_seen", "val_unseen"):
        metric = selected_metrics.get(split, {}).get("by_extent", {}).get(selected_extent, {})
        for pool_name, pool in metric.get("negative_pools", {}).items():
            lines.append(f"| {split} | {pool_name} | {pool.get('n', 0)} | {fmt(pool.get('accuracy'))} | {fmt(pool.get('auroc'))} | {fmt(pool.get('margin'))} |")
        confounds = metric.get("distance_confounds", {})
        lines.append(f"| {split} | distance adjusted for altitude + brightness + map | {confounds.get('n', 0)} | — | — | partial ρ={fmt(confounds.get('partial_distance_rho'))} (p={fmt(confounds.get('partial_distance_pvalue'), 3)}) |")
        queries = load_jsonl(args.dataset_dir / f"queries_{split}.jsonl")
        templates = [r for r in load_jsonl(args.dataset_dir / f"templates_{split}.jsonl")
                     if int(r["extent_m"]) == int(selected_extent)]
        scene_by_map = {}
        for row in templates:
            scene_by_map.setdefault(row["map_name"], set()).add(row["scene_key"])
        rank_chance = [1.0 / len(scene_by_map[q["map_name"]]) for q in queries
                       if len(scene_by_map.get(q["map_name"], ())) > 1]
        random_rank = float(np.mean(rank_chance)) if rank_chance else None
        hard = metric.get("negative_pools", {}).get("same_map", {})
        lines.append(f"| {split} | uniform same-map random-rank baseline | {len(rank_chance)} | {fmt(random_rank)} | — | observed hard-negative accuracy={fmt(hard.get('accuracy'))} |")
    lines += [
        "",
        "Hard-negative pools are explicitly separated into random, same-map, nearby (≤200 m), and visually similar. The preregistered Gate 1 threshold is same-map accuracy ≥0.55, positive margin, paired sign test p<0.05, and a negative query-to-goal distance Spearman trend (p<0.05) on val_seen; the selected encoder/extent is then reported on untouched val_unseen. Similarity-vs-distance values are query-to-positive-template cosine, normalized per episode; distance confounds also report residual Spearman after altitude, brightness, and map fixed effects.",
        "",
        "Interpretation: the selected partial-tuned SigLIP2 is above a uniform same-map random-rank baseline (0.049 on val_seen; 0.008 on val_unseen), so the result is not equivalent to no visual information. It also shows a strong approach trend (val_seen ρ=-0.355; adjusted ρ=-0.244 after altitude, brightness, and map fixed effects). However, its strongest same-map distractor still beats the true scene on most queries (accuracy 0.220; margin -0.022; AUROC 0.725 over the full same-map negative pool), and held-out accuracy is only 0.081. Under the preregistered reliability gate, this is insufficient for dependable place identity matching.",
        "",
        "![Full RGB similarity by distance stratum](artifacts/visual_goal_abstraction/eval/similarity_vs_distance.png)",
        "",
        "## B. Progressive abstraction L0–L9",
        "",
    ]
    abstraction = data.get("abstraction")
    if not abstraction:
        lines += ["Gate 1 did not pass (or abstraction was not requested); per protocol the abstraction, ablations, cross-view retrieval, Top-K reranking, and oracle stages were stopped.", ""]
    else:
        lines += [
            "| Level | Split | R@1 | R@5 | R@10 | Same-map accuracy | Same-map margin | Distance ρ |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for split in ("val_seen", "val_unseen"):
            for level in [f"L{i}" for i in range(10)]:
                metric = abstraction.get(split, {}).get(level)
                if metric:
                    hard = metric["negative_pools"]["same_map"]
                    lines.append(f"| {level} | {split} | {fmt(metric.get('R@1'))} | {fmt(metric.get('R@5'))} | {fmt(metric.get('R@10'))} | {fmt(hard.get('accuracy'))} | {fmt(hard.get('margin'))} | {fmt(metric.get('distance_spearman_rho'))} |")
        for split in ("val_seen", "val_unseen"):
            level_data = abstraction.get(split, {})
            base = level_data.get("L0", {})
            base_acc = base.get("negative_pools", {}).get("same_map", {}).get("accuracy")
            base_r1 = base.get("R@1")
            sufficient = None
            for i in range(10):
                m = level_data.get(f"L{i}")
                if not m or base_r1 is None or base_acc is None:
                    continue
                acc = m.get("negative_pools", {}).get("same_map", {}).get("accuracy")
                if m.get("R@1", 0) >= .90 * base_r1 and acc is not None and acc >= base_acc - .05:
                    sufficient = f"L{i}"
                    break
            lines += ["", f"Lowest level meeting the preregistered dual criterion on {split}: **{sufficient or 'none'}** (R@1 ≥ 90% of L0 and same-map accuracy within 5 percentage points of L0). The selected level is not moved using val_unseen; this split is reported as held out."]
        lines += [""]
    lines += [
        "### What each level removes",
        "",
        "| Level | Texture | Color | Background | Semantic regions | Geometry/layout |",
        "|---|---|---|---|---|---|",
        "| L0 Full RGB | ✓ | ✓ | ✓ | implicit | ✓ |",
        "| L1 Background Blur | local preserved | ✓ | surrounding blurred | implicit | ✓ |",
        "| L2 Target + Anchor | retained locally | ✓ | removed outside masks | implicit | ✓ |",
        "| L3 Low-frequency | removed | coarse | ✓ | implicit | ✓ |",
        "| L4 Posterized | ✓ | 4/8/16-level quantized | ✓ | implicit | ✓ |",
        "| L5 Grayscale | ✓ | removed | ✓ | implicit | ✓ |",
        "| L6 Semantic Regions | removed | fixed palette | coarse | ✓ | ✓ |",
        "| L7 Contour + Color | removed | coarse | removed | partial | ✓ |",
        "| L8 Pure Contour | removed | removed | removed | removed | ✓ |",
        "| L9 Target + Anchor Geometry | removed | removed | removed | target/anchor only | ✓ |",
        "",
        "The full machine-readable result (including all distance bins, per-episode curves, negative pools, and confound controls) is `artifacts/visual_goal_abstraction/eval/full_goal_retrieval.json`.",
        "",
        "## C. Minimal sufficient representation",
        "",
        "**Not determined.** Full RGB itself failed same-map retrieval, so the protocol stopped before testing whether texture, color, target appearance, anchor appearance, geometry, or context is minimally sufficient. No abstraction level is claimed to work.",
        "",
        "## D. Target vs anchor vs context",
        "",
        "**Not run by the Gate 1 stop rule.** Target-only, anchor-only, target+anchor, and context contribution are not inferred from the full-RGB result. The implementation is ready to run only after a future valid full-template matching gate.",
        "",
    ]
    controls = data.get("controlled_ablation", {})
    if controls:
        lines += [
            "| Controlled goal template | Split | R@1 | R@5 | Same-map accuracy | Margin |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name in ("target_only", "anchor_only", "target_anchor", "target_anchor_context",
                     "color_full", "color_gray", "color_8", "color_none",
                     "texture_full", "texture_blur", "texture_strong_blur", "texture_none",
                     "geometry_full", "geometry_contour", "geometry_coarse_footprint", "geometry_none",
                     "context_no_road", "context_no_parking", "context_no_buildings",
                     "context_no_vegetation", "context_no_open_area"):
            for split in ("val_seen", "val_unseen"):
                metric = controls.get(split, {}).get(name)
                if metric:
                    hard = metric.get("negative_pools", {}).get("same_map", {})
                    lines.append(f"| {name} | {split} | {fmt(metric.get('R@1'))} | {fmt(metric.get('R@5'))} | {fmt(hard.get('accuracy'))} | {fmt(hard.get('margin'))} |")
        lines.append("")
    lines += [
        "## E. Cross-view",
        "",
        "True oblique and FPV comparisons are unavailable: CityNav supplies no captured frames and HETT has no calibrated 3D scene/camera renderer. We cannot determine whether viewpoint gap is the main bottleneck. The altitude/brightness/map-adjusted distance correlation is a confound control within the orthographic interface, not a cross-view test.",
        "",
        "## F. Belief Top-K reranking and G. Oracle upper bound",
        "",
        "**Not run:** Gate 1 failed, so no B0 Top-K visual rerank or oracle was evaluated. Thus there is no measured R@1 improvement and no empirical oracle upper bound from this run. The candidate reranker/exporter remains available for a future experiment if the full-template gate is first made reliable.",
        "",
        "## Case judgement",
        "",
    ]
    if not gate.get("pass"):
        case = "**CASE D** — Full RGB Goal Template retrieval failed the same-map validation-seen gate. Stop visual-imagination/diffusion work under this protocol."
    elif abstraction:
        seen_levels = abstraction.get("val_seen", {})
        base_r1 = seen_levels.get("L0", {}).get("R@1", 0)
        base_acc = seen_levels.get("L0", {}).get("negative_pools", {}).get("same_map", {}).get("accuracy", 0)
        sufficient = any(
            seen_levels.get(f"L{i}", {}).get("R@1", 0) >= .90 * base_r1 and
            seen_levels.get(f"L{i}", {}).get("negative_pools", {}).get("same_map", {}).get("accuracy", -1) >= base_acc - .05
            for i in range(1, 10))
        case = "**CASE A** — Full RGB and an abstraction meet the predeclared dual sufficiency criterion." if sufficient else "**CASE B** — Full RGB works, but the tested abstractions lose too much retrieval performance."
    else:
        case = "**Gate 1 passed; CASE A/B/C awaits the requested abstraction/cross-view runs.**"
    lines += [case, "", "## Viewable examples", ""]
    for split in ("val_seen", "val_unseen"):
        gate_path = result_dir / "visualizations" / f"full_rgb_gate_examples_{split}_n100.jpg"
        if gate_path.exists():
            lines.append(f"- [{split}: 100 full-RGB query / goal / same-map-hard-negative panels](artifacts/visual_goal_abstraction/eval/visualizations/{gate_path.name})")
        path = result_dir / "visualizations" / f"visual_examples_{split}_n100.jpg"
        if path.exists():
            lines.append(f"- [{split}: 100 query/template examples](artifacts/visual_goal_abstraction/eval/visualizations/{path.name})")
    lines += ["", "## Reproduction", "", "Run from the repository root:", "", "```bash", "bash multiagent/scripts/run_visual_goal_experiment.sh", "```", ""]
    args.output.write_text("\n".join(lines))
    _plots(data, result_dir)
    print(args.output)


def _plots(data, out_dir):
    abstraction = data.get("abstraction", {})
    out_dir.mkdir(parents=True, exist_ok=True)
    if abstraction:
        fig, ax = plt.subplots(figsize=(10, 5))
        for split in ("val_seen", "val_unseen"):
            metrics = abstraction.get(split, {})
            levels = [metrics.get(f"L{i}", {}) for i in range(10)]
            y = [x.get("negative_pools", {}).get("same_map", {}).get("accuracy", np.nan) for x in levels]
            ax.plot(range(10), y, marker="o", label=f"{split} same-map acc")
            y = [x.get("R@1", np.nan) for x in levels]
            ax.plot(range(10), y, marker="s", linestyle="--", label=f"{split} R@1")
        ax.set(xticks=range(10), xlabel="Goal-template abstraction level", ylabel="Retrieval metric", ylim=(0, 1), title="Performance vs. visual abstraction")
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "performance_vs_abstraction.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [">80m", "40-80m", "20-40m", "0-20m"]
    for split in ("val_seen", "val_unseen"):
        selected = data.get("selected_on_val_seen_only", {}).get("encoder")
        extent = str(data.get("selected_on_val_seen_only", {}).get("extent_m", "80"))
        base = data.get("models", {}).get(selected, {}).get(split, {}).get("by_extent", {}).get(extent, {}).get("similarity_by_distance", {})
        y = [base.get(label, {}).get("mean", np.nan) for label in labels]
        ax.plot(labels, y, marker="o", label=split)
    ax.set(xlabel="UAV distance to GT goal", ylabel="Cosine similarity", title="Full RGB similarity by distance stratum")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "similarity_vs_distance.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
