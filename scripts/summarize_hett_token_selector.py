"""Aggregate the three pre-registered relational heatmap seeds."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = [ROOT / "runs" / f"hett_token_selector_s{index}" for index in range(3)]


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": round(float(values.mean()), 4), "std": round(float(values.std(ddof=1)), 4),
            "values": values.tolist()}


def main():
    results = [json.load((run / "results.json").open()) for run in RUNS]
    aggregate = {"seeds": [result["seed"] for result in results]}
    for split in ["val_seen", "val_unseen"]:
        aggregate[split] = {}
        for metric in ["top1_hit20", "top5_recall20", "median_top1_m", "mean_mass20"]:
            aggregate[split][metric] = stats([result[split]["learned"][metric] for result in results])
        aggregate[split]["by_relation"] = {}
        for relation in results[0][split]["by_relation"]:
            aggregate[split]["by_relation"][relation] = {
                metric: stats([result[split]["by_relation"][relation][metric] for result in results])
                for metric in ["top1_hit20", "top5_recall20"]
            }
    unseen = aggregate["val_unseen"]
    seen = aggregate["val_seen"]
    passed = (unseen["top5_recall20"]["mean"] >= 70 and unseen["top1_hit20"]["mean"] >= 35
              and seen["top1_hit20"]["mean"] - unseen["top1_hit20"]["mean"] <= 15)
    aggregate["passed"] = passed
    output = ROOT / "runs" / "hett_token_selector_aggregate.json"
    output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2))

    lines = ["# HETT token-mean 几何候选选择器：三 seed 汇总", "",
             "| 划分 | Top-1 Hit@20 | Top-5 Recall@20 | 中位误差 | 20m 概率质量 |",
             "|---|---:|---:|---:|---:|"]
    for split in ["val_seen", "val_unseen"]:
        row = aggregate[split]
        lines.append(f"| {split} | {row['top1_hit20']['mean']:.2f}±{row['top1_hit20']['std']:.2f}% | "
                     f"{row['top5_recall20']['mean']:.2f}±{row['top5_recall20']['std']:.2f}% | "
                     f"{row['median_top1_m']['mean']:.2f}±{row['median_top1_m']['std']:.2f} m | "
                     f"{row['mean_mass20']['mean']:.3f}±{row['mean_mass20']['std']:.3f} |")
    lines += ["", "## val_unseen 分关系", "",
              "| 类型 | Top-1 Hit@20 | Top-5 Recall@20 |", "|---|---:|---:|"]
    for relation, row in aggregate["val_unseen"]["by_relation"].items():
        lines.append(f"| {relation} | {row['top1_hit20']['mean']:.2f}±{row['top1_hit20']['std']:.2f}% | "
                     f"{row['top5_recall20']['mean']:.2f}±{row['top5_recall20']['std']:.2f}% |")
    lines += ["", "## 判定", "",
              "**通过。**" if passed else "**未通过。**",
              "", "模型稳定优于地标中心基线，但未达到预注册的 unseen 35% Top-1 / 70% Top-5 门槛。"]
    (ROOT / "runs" / "HETT_TOKEN_SELECTOR_AGGREGATE.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    relations = list(aggregate["val_unseen"]["by_relation"])
    top1 = [aggregate["val_unseen"]["by_relation"][key]["top1_hit20"]["mean"] for key in relations]
    top5 = [aggregate["val_unseen"]["by_relation"][key]["top5_recall20"]["mean"] for key in relations]
    x = np.arange(len(relations)); width = .38
    ax.bar(x - width / 2, top1, width, label="Top-1")
    ax.bar(x + width / 2, top5, width, label="Top-5")
    ax.axhline(70, color="gray", linestyle="--", linewidth=1, label="Top-5 gate")
    ax.set_xticks(x, relations, rotation=25, ha="right")
    ax.set_ylabel("Hit / Recall @20 (%)"); ax.set_title("val_unseen by relation, mean of 3 seeds")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "runs" / "hett_token_selector_relations.png", dpi=180)


if __name__ == "__main__":
    main()
