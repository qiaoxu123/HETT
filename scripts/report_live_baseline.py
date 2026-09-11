"""Create a read-only baseline health/loss report from supervised run artifacts."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


WEIGHTS = {'direction': 1.5, 'progress': 0.1, 'goal': 2.0, 'target': 0.1}


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


def loss_breakdown(row):
    raw = {
        'direction': float(row['direction_loss']),
        'progress': float(row['progress_loss']),
        'goal': float(row['goal_loss']),
    }
    known = sum(WEIGHTS[name] * value for name, value in raw.items())
    target_weighted = float(row['il_loss']) - known
    raw['target'] = target_weighted / WEIGHTS['target']
    weighted = {name: raw[name] * WEIGHTS[name] for name in WEIGHTS}
    reconstructed = sum(weighted.values())
    if not all(math.isfinite(value) for value in (*raw.values(), *weighted.values())):
        raise ValueError(f'non-finite loss row: {row}')
    if target_weighted < -1e-6 or abs(reconstructed - float(row['il_loss'])) > 1e-6:
        raise ValueError(f'inconsistent loss decomposition: {row}')
    return {
        'raw': raw,
        'weighted': weighted,
        'share_percent': {name: 100 * value / reconstructed
                          for name, value in weighted.items()},
        'reconstructed_il_loss': reconstructed,
    }


def gradient_health(rows):
    result = {}
    for epoch in sorted({row['epoch'] for row in rows}):
        values = np.asarray([row['grad_norm'] for row in rows if row['epoch'] == epoch], dtype=float)
        losses = np.asarray([row['recent_il_loss'] for row in rows if row['epoch'] == epoch], dtype=float)
        result[str(epoch)] = {
            'samples': len(values),
            'all_finite': bool(np.isfinite(values).all() and np.isfinite(losses).all()),
            'grad_p50': float(np.percentile(values, 50)),
            'grad_p95': float(np.percentile(values, 95)),
            'grad_p99': float(np.percentile(values, 99)),
            'grad_max': float(values.max()),
            'above_clip_40': int((values > 40).sum()),
            'above_100': int((values > 100).sum()),
            'loss_p95': float(np.percentile(losses, 95)),
            'loss_max': float(losses.max()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epochs = json_lines(args.run_dir / 'checkpoints/epoch_metrics.jsonl')
    batches = json_lines(args.run_dir / 'checkpoints/batch_metrics.jsonl')
    status = json.loads((args.run_dir / 'status.json').read_text())
    completed = []
    for row in epochs:
        item = {'epoch': row['epoch'], 'elapsed_seconds': row['elapsed_seconds'],
                'loss': loss_breakdown(row), 'validation': row['validation']}
        completed.append(item)

    report = {
        'run_dir': str(args.run_dir.resolve()),
        'status': status,
        'completed_epochs': completed,
        'latest_batch': batches[-1] if batches else None,
        'gradient_health': gradient_health(batches),
        'method': ('target raw loss = (IL - 1.5*direction - 0.1*progress - 2.0*goal) / 0.1; '
                   'weighted loss share is a scale diagnostic, not causal importance; '
                   'logged gradient norm is ET-only before clipping at 40 and does not measure '
                   'the separate language/vision optimizer gradients'),
    }
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#f8fafc', 'axes.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {'direction': '#2563eb', 'progress': '#e58b21',
              'goal': '#169c78', 'target': '#9c50bf'}
    if completed:
        x = np.array([item['epoch'] for item in completed])
        bottom = np.zeros(len(completed))
        for name in WEIGHTS:
            values = np.array([item['loss']['share_percent'][name] for item in completed])
            axes[0, 0].bar(x, values, bottom=bottom, color=colors[name], label=name)
            bottom += values
        axes[0, 0].set(title='Weighted loss composition', xlabel='epoch', ylabel='percent', ylim=(0, 100))
        axes[0, 0].legend(fontsize=8)
        for split, style in [('val_seen', '--'), ('val_unseen', '-')]:
            axes[0, 1].plot(x, [item['validation'][split]['sr'] for item in completed],
                            marker='o', linestyle=style, label=f'{split} SR')
            axes[0, 1].plot(x, [item['validation'][split]['spl'] for item in completed],
                            marker='s', linestyle=style, label=f'{split} SPL')
        axes[0, 1].set(title='Closed-loop validation', xlabel='epoch', ylabel='percent')
        axes[0, 1].legend(fontsize=8)
    for epoch in sorted({row['epoch'] for row in batches}):
        rows = [row for row in batches if row['epoch'] == epoch]
        axes[1, 0].plot([row['batch'] for row in rows], [row['recent_il_loss'] for row in rows],
                        alpha=.8, label=f'epoch {epoch}')
        axes[1, 1].plot([row['batch'] for row in rows], [row['grad_norm'] for row in rows],
                        alpha=.8, label=f'epoch {epoch}')
    axes[1, 0].set(title='Sampled training loss', xlabel='micro-batch', ylabel='IL loss')
    axes[1, 1].set(title='ET gradient norm before clipping', xlabel='micro-batch', ylabel='norm', yscale='log')
    axes[1, 1].axhline(40, color='#dc2626', linestyle='--', linewidth=1, label='clip=40')
    for axis in axes.flat:
        axis.grid(alpha=.15)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.suptitle('Original baseline: live read-only report')
    fig.savefig(args.output_dir / 'overview.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'completed_epochs': len(completed),
                      'batch_samples': len(batches),
                      'output': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
