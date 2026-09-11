"""Aggregate loss, validation curves, runtime and GPU usage across full training seeds."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LOSS_NAMES = ('direction', 'progress', 'goal', 'target')
WEIGHT_KEYS = {name: f'{name}_loss_weight' for name in LOSS_NAMES}


def json_lines(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def weighted_record(row, arguments):
    weights = {name: float(arguments[WEIGHT_KEYS[name]]) for name in LOSS_NAMES}
    raw = {name: float(row[f'{name}_loss']) for name in LOSS_NAMES if f'{name}_loss' in row}
    if 'target' not in raw:
        known = sum(weights[name] * raw[name] for name in ('direction', 'progress', 'goal'))
        raw['target'] = ((float(row['il_loss']) - known) / weights['target']
                         if weights['target'] else 0.0)
    weighted = {name: weights[name] * raw[name] for name in LOSS_NAMES}
    total = float(row['il_loss'])
    reconstructed = sum(weighted.values())
    if not all(math.isfinite(value) for value in [total, reconstructed, *raw.values()]):
        raise ValueError(f'non-finite loss record: {row}')
    if abs(total - reconstructed) > 1e-4:
        raise ValueError(f'loss components do not reconstruct IL loss: {total} vs {reconstructed}')
    return {
        'epoch': int(row['epoch']), 'il_loss': total, 'raw': raw, 'weights': weights,
        'weighted': weighted,
        'weighted_share_percent': {
            name: (100 * value / reconstructed if reconstructed else 0.0)
            for name, value in weighted.items()},
        'elapsed_seconds': float(row['elapsed_seconds']),
        'validation': row['validation'],
    }


def load_training_run(path):
    checkpoint = path / 'checkpoints'
    arguments = json.loads((checkpoint / 'training_args.json').read_text())
    rows = [weighted_record(row, arguments)
            for row in json_lines(checkpoint / 'epoch_metrics.jsonl')]
    if not rows:
        raise FileNotFoundError(f'no epoch metrics under {path}')
    telemetry = json_lines(path / 'telemetry.jsonl')
    memory_mib = []
    for row in telemetry:
        try:
            memory_mib.append(float(row['gpu'].splitlines()[0].split(',')[1].strip()))
        except (KeyError, IndexError, ValueError):
            pass
    epoch_seconds = np.diff([0, *[row['elapsed_seconds'] for row in rows]])
    best = max(rows, key=lambda row: row['validation']['val_unseen']['sr'])
    return {
        'epochs': rows,
        'summary': {
            'completed_epochs': len(rows),
            'mean_epoch_seconds': float(epoch_seconds.mean()),
            'total_training_seconds': rows[-1]['elapsed_seconds'],
            'peak_gpu_memory_gib': max(memory_mib) / 1024 if memory_mib else None,
            'best_val_unseen_sr': float(best['validation']['val_unseen']['sr']),
            'best_val_unseen_epoch': best['epoch'],
            'final_val_unseen_sr': float(rows[-1]['validation']['val_unseen']['sr']),
            'final_val_unseen_spl': float(rows[-1]['validation']['val_unseen']['spl']),
            'final_val_unseen_ne': float(rows[-1]['validation']['val_unseen']['ne']),
            'final_weighted_loss_share_percent': rows[-1]['weighted_share_percent'],
        },
    }


def curve_statistics(runs, getter):
    common_epochs = sorted(set.intersection(*(
        {row['epoch'] for row in run['epochs']} for run in runs.values())))
    values = []
    for seed in sorted(runs):
        indexed = {row['epoch']: row for row in runs[seed]['epochs']}
        values.append([getter(indexed[epoch]) for epoch in common_epochs])
    values = np.asarray(values, dtype=float)
    return {'epochs': common_epochs, 'mean': values.mean(axis=0).tolist(),
            'std': values.std(axis=0, ddof=1).tolist() if len(values) > 1
            else np.zeros(len(common_epochs)).tolist(),
            'per_seed': {str(seed): row for seed, row in zip(sorted(runs), values.tolist())}}


def aggregate(loaded):
    result = {}
    for name, runs in loaded.items():
        result[name] = {
            'seeds': {str(seed): run['summary'] for seed, run in sorted(runs.items())},
            'curves': {
                'il_loss': curve_statistics(runs, lambda row: row['il_loss']),
                'val_unseen_sr': curve_statistics(
                    runs, lambda row: row['validation']['val_unseen']['sr']),
                'val_unseen_spl': curve_statistics(
                    runs, lambda row: row['validation']['val_unseen']['spl']),
                'val_unseen_ne': curve_statistics(
                    runs, lambda row: row['validation']['val_unseen']['ne']),
            },
            'final_weighted_components': {
                loss: [runs[seed]['epochs'][-1]['weighted'][loss] for seed in sorted(runs)]
                for loss in LOSS_NAMES},
        }
    return result


def plot(result, output_dir):
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(result))))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    specs = [('il_loss', 'IL loss', False), ('val_unseen_sr', 'Val-unseen SR', False),
             ('val_unseen_spl', 'Val-unseen SPL', False),
             ('val_unseen_ne', 'Val-unseen NE', True)]
    for axis, (metric, title, lower) in zip(axes.flat, specs):
        for index, (name, values) in enumerate(result.items()):
            curve = values['curves'][metric]
            x = np.asarray(curve['epochs'])
            mean = np.asarray(curve['mean'])
            std = np.asarray(curve['std'])
            line_color = colors[index]
            axis.plot(x, mean, label=name, color=line_color)
            axis.fill_between(x, mean - std, mean + std, alpha=.12, color=line_color)
        axis.set(title=f'{title} (mean ± seed SD)', xlabel='epoch')
        axis.grid(alpha=.15)
        axis.legend(fontsize=7, ncol=2)
        if lower:
            axis.set_ylabel('meters; lower is better')
    fig.savefig(output_dir / 'training_curves.png', dpi=180)
    plt.close(fig)

    names = list(result)
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    fig, axis = plt.subplots(figsize=(max(11, len(names) * 1.35), 5), constrained_layout=True)
    palette = {'direction': '#2563eb', 'progress': '#d97706',
               'goal': '#059669', 'target': '#9333ea'}
    for loss in LOSS_NAMES:
        values = np.asarray([np.mean(result[name]['final_weighted_components'][loss])
                             for name in names])
        axis.bar(x, values, bottom=bottom, label=loss, color=palette[loss])
        bottom += values
    axis.set(title='Final weighted loss composition (seed mean)', ylabel='weighted loss')
    axis.set_xticks(x, names, rotation=25, ha='right')
    axis.legend()
    axis.grid(axis='y', alpha=.15)
    fig.savefig(output_dir / 'weighted_loss_components.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME:SEED=/absolute/run/path')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    loaded = {}
    paths = {}
    for specification in args.run:
        identity, path_text = specification.split('=', 1)
        name, seed_text = identity.rsplit(':', 1)
        seed = int(seed_text)
        path = Path(path_text)
        loaded.setdefault(name, {})[seed] = load_training_run(path)
        paths.setdefault(name, {})[str(seed)] = str(path.resolve())
    result = aggregate(loaded)
    output = {'paths': paths, 'runs': result,
              'note': 'curves show mean ± seed SD; loss scale is not causal importance'}
    (args.output_dir / 'summary.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    plot(result, args.output_dir)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
