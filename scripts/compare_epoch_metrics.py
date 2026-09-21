#!/usr/bin/env python3
"""Compare a short candidate run with matching baseline epochs and baseline best."""
import argparse
import json
from pathlib import Path


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def number(value):
    return f"{value:.2f}"


def change(value):
    return f"{value:+.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = load_rows(args.candidate)
    baseline = load_rows(args.baseline)
    if not candidate:
        raise SystemExit("candidate metrics are empty")
    baseline_by_epoch = {row["epoch"]: row for row in baseline}
    missing = [row["epoch"] for row in candidate if row["epoch"] not in baseline_by_epoch]
    if missing:
        raise SystemExit(f"baseline is missing epochs: {missing}")

    candidate_best = max(
        candidate, key=lambda row: row["validation"]["val_unseen"]["sr"]
    )
    baseline_best = max(
        baseline, key=lambda row: row["validation"]["val_unseen"]["sr"]
    )
    lines = [
        "# Teacher trajectory experiment comparison",
        "",
        "## Matched training epochs",
        "",
        "| epoch | split | candidate SR | baseline SR | ΔSR | candidate SPL | baseline SPL | ΔSPL | candidate NE | baseline NE | ΔNE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate:
        reference = baseline_by_epoch[row["epoch"]]
        for split in ("val_seen", "val_unseen"):
            left = row["validation"][split]
            right = reference["validation"][split]
            lines.append(
                f"| {row['epoch']} | {split} | {number(left['sr'])} | {number(right['sr'])} | "
                f"{change(left['sr'] - right['sr'])} | {number(left['spl'])} | "
                f"{number(right['spl'])} | {change(left['spl'] - right['spl'])} | "
                f"{number(left['ne'])} | {number(right['ne'])} | "
                f"{change(left['ne'] - right['ne'])} |"
            )

    lines += [
        "",
        "NE is lower-is-better; SR and SPL are higher-is-better.",
        "",
        "## Training losses",
        "",
        "| epoch | IL loss | direction loss | progress loss | goal loss |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in candidate:
        lines.append(
            f"| {row['epoch']} | {number(row['il_loss'])} | "
            f"{number(row['direction_loss'])} | {number(row['progress_loss'])} | "
            f"{number(row['goal_loss'])} |"
        )

    lines += [
        "",
        "## Candidate best versus full baseline best",
        "",
        f"Candidate epoch: {candidate_best['epoch']}; baseline epoch: {baseline_best['epoch']}.",
        "",
        "| split | candidate SR | baseline SR | ΔSR | candidate SPL | baseline SPL | ΔSPL | candidate NE | baseline NE | ΔNE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("val_seen", "val_unseen"):
        left = candidate_best["validation"][split]
        right = baseline_best["validation"][split]
        lines.append(
            f"| {split} | {number(left['sr'])} | {number(right['sr'])} | "
            f"{change(left['sr'] - right['sr'])} | {number(left['spl'])} | "
            f"{number(right['spl'])} | {change(left['spl'] - right['spl'])} | "
            f"{number(left['ne'])} | {number(right['ne'])} | "
            f"{change(left['ne'] - right['ne'])} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
