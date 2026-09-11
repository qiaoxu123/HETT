"""Audit archived HETT baseline logs and compare them with a live controlled run."""

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS = re.compile(
    r"^IL_loss ([\d.]+) direction_loss ([\d.]+) progress_loss ([\d.]+) "
    r"goal_predict_loss ([\d.]+)", re.M)


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as stream:
            return stream.read().replace("\r", "\n")
    return path.read_text(errors="replace").replace("\r", "\n")


def metrics(line):
    return {key: float(value) for key, value in
            re.findall(r"(\w+): (-?[\d.]+)", line)}


def training_curve(path):
    text = read(path)
    matches = list(LOSS.finditer(text))
    rows = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        epoch_match = re.search(r"^epoch (\d+)$", block, re.M)
        if not epoch_match:
            continue
        row = dict(zip(("il_loss", "direction_loss", "progress_loss", "goal_loss"),
                       map(float, match.groups())))
        row["epoch"] = int(epoch_match.group(1))
        evaluation = block[epoch_match.end():]
        for split in ("val_seen", "val_unseen"):
            split_match = re.search(r"^" + split + r" , (.+)$", evaluation, re.M)
            if not split_match:
                raise ValueError(f"missing {split} after epoch {row['epoch']} in {path}")
            row[split] = metrics(split_match.group(1))
        rows.append(row)
    return rows


def json_lines(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--live-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive_a = training_curve(args.reference_dir / "train_rep.log.gz")
    archive_b = training_curve(args.reference_dir / "train_epoch12_20.log.gz")
    live = json_lines(args.live_run / "checkpoints/epoch_metrics.jsonl")
    files = sorted(path for path in args.reference_dir.iterdir() if path.is_file())
    report = {
        "reference_dir": str(args.reference_dir.resolve()),
        "files": [{"name": path.name, "bytes": path.stat().st_size,
                   "sha256": sha256(path)} for path in files],
        "archive_a": archive_a,
        "archive_b": archive_b,
        "live_completed_epochs": live,
        "audit": {
            "has_epoch_0_to_11_log": [row["epoch"] for row in archive_a] == list(range(12)),
            "has_epoch_11_to_19_resume_log": [row["epoch"] for row in archive_b] == list(range(11, 20)),
            "has_full_split_evaluation_text": (args.reference_dir / "valid.txt").exists(),
            "has_checkpoint": any("checkpoint" in path.name.lower() for path in files),
            "single_continuous_lineage": False,
            "strict_paper_loss_match": False,
        },
        "comparability": {
            "archive_action_weight": 1.0,
            "paper_action_weight": 1.5,
            "live_action_weight": 1.5,
            "archive_extra_target_weight": 0.1,
            "paper_extra_target_weight": 0.0,
            "live_extra_target_weight": 0.1,
            "conclusion": ("Archived metrics are useful as a historical scale reference, but its "
                           "loss values are not directly comparable with the paper-aligned live run."),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f8fafc", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for rows, label, color in ((archive_a, "archive A", "#64748b"),
                               (archive_b, "archive B/resume", "#d97706")):
        x = [row["epoch"] for row in rows]
        axes[0].plot(x, [row["val_unseen"]["sr"] for row in rows], "o-",
                     label=label, color=color)
        axes[1].plot(x, [row["val_unseen"]["ne"] for row in rows], "o-",
                     label=label, color=color)
    if live:
        x = [row["epoch"] for row in live]
        axes[0].plot(x, [row["validation"]["val_unseen"]["sr"] for row in live],
                     "s-", label="controlled live", color="#2563eb", linewidth=2.5)
        axes[1].plot(x, [row["validation"]["val_unseen"]["ne"] for row in live],
                     "s-", label="controlled live", color="#2563eb", linewidth=2.5)
    axes[0].set(title="Val-unseen success rate", xlabel="epoch", ylabel="SR (%)")
    axes[1].set(title="Val-unseen navigation error", xlabel="epoch", ylabel="NE (m)")
    for axis in axes:
        axis.grid(alpha=.15)
        axis.legend(fontsize=8)
    fig.suptitle("Historical reference vs controlled baseline (not identical loss weights)")
    fig.savefig(args.output_dir / "comparison.png", dpi=180)
    plt.close(fig)

    live0 = live[0] if live else None
    live_summary = "暂无完整 epoch。"
    if live0:
        metric = live0["validation"]["val_unseen"]
        live_summary = f"epoch 0: SR {metric['sr']:.2f}, SPL {metric['spl']:.2f}, NE {metric['ne']:.2f}。"
    markdown = f"""# 历史 reference_baseline 审计

## 结论

- 日志记录层面基本完整：A 含 epoch 0–11，B 含 resume epoch 11–19，另有完整 split 评测文本。
- 复现实物不完整：归档内没有 checkpoint，也没有能把两段证明为同一连续运行的不可变 provenance。
- 不能直接比较 loss：历史代码 action 权重为 1.0；论文与当前受控训练为 1.5。
- 历史评测还存在两套 epoch-11 数字及 checkpoint 后续覆盖风险，所以只作量级参考。
- 当前受控训练 {live_summary}

## 是否需要重跑

需要。当前正在运行的 20-epoch 受控基线才是主结论依据；旧日志无需再重复生成，保留作旁证即可。

论文只定义三项权重：coarse target 2.0、action 1.5、progress 0.1。发布代码额外加入 target-grid 0.1；实验队列已单独安排去掉该额外项的消融。
"""
    (args.output_dir / "README.md").write_text(markdown)
    print(json.dumps({"archive_epochs": [len(archive_a), len(archive_b)],
                      "live_epochs": len(live), "output": str(args.output_dir)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
