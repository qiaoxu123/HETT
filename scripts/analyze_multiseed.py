"""Aggregate full HETT runs across seeds without selecting the best seed."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_validation_matrix import load_run, paired_summary


T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def statistics(values):
    values = np.asarray(values, dtype=float)
    count = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    critical = T_975.get(count, 1.96)
    half = critical * std / np.sqrt(count) if count > 1 else 0.0
    return {'n': count, 'mean': mean, 'std': std,
            'ci95': [mean - half, mean + half], 'values': values.tolist()}


def aggregate(loaded, pairs):
    result = {'runs': {}, 'pairs': {}}
    for name, seeds in loaded.items():
        result['runs'][name] = {}
        splits = sorted(set.intersection(*(set(run) for run in seeds.values())))
        for split in splits:
            metric_names = sorted(set.intersection(*(
                {key for key, value in run[split]['reported'].items()
                 if isinstance(value, (int, float))} for run in seeds.values())))
            result['runs'][name][split] = {
                metric: statistics([seeds[seed][split]['reported'][metric]
                                    for seed in sorted(seeds)])
                for metric in metric_names
            }
    for left, right in pairs:
        pair_name = f'{left}:{right}'
        result['pairs'][pair_name] = {}
        common_seeds = sorted(set(loaded[left]) & set(loaded[right]))
        common_splits = sorted(set.intersection(*(
            set(loaded[left][seed]) & set(loaded[right][seed]) for seed in common_seeds)))
        for split in common_splits:
            per_seed = {
                str(seed): paired_summary(loaded[left][seed][split]['episodes'],
                                          loaded[right][seed][split]['episodes'])
                for seed in common_seeds
            }
            result['pairs'][pair_name][split] = {
                'per_seed': per_seed,
                'success_delta_pp': statistics([
                    value['success_delta_pp'] for value in per_seed.values()]),
                'map_balanced_success_delta_pp': statistics([
                    value['map_balanced_success_delta_pp'] for value in per_seed.values()]),
                'mean_final_distance_delta_m': statistics([
                    value['mean_final_distance_delta_m'] for value in per_seed.values()]),
                'success_wins_total': sum(value['success_wins'] for value in per_seed.values()),
                'success_losses_total': sum(value['success_losses'] for value in per_seed.values()),
            }
    return result


def plot(result, output_dir):
    names = list(result['runs'])
    available = [name for name in names if 'val_unseen' in result['runs'][name]]
    x = np.arange(len(available))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(available) * 1.2), 8),
                             constrained_layout=True)
    for axis, metric, title in [(axes[0], 'sr', 'Val-unseen success rate'),
                                (axes[1], 'spl', 'Val-unseen SPL')]:
        means = [result['runs'][name]['val_unseen'][metric]['mean'] for name in available]
        stds = [result['runs'][name]['val_unseen'][metric]['std'] for name in available]
        axis.bar(x, means, yerr=stds, capsize=4)
        axis.set_ylabel('percent')
        axis.set_title(f'{title}: mean ± seed standard deviation')
        axis.set_xticks(x, available, rotation=30, ha='right')
    fig.savefig(output_dir / 'multiseed_metrics.png', dpi=180)

    pair_names = list(result['pairs'])
    means = [result['pairs'][name]['val_unseen']['map_balanced_success_delta_pp']['mean']
             for name in pair_names]
    stds = [result['pairs'][name]['val_unseen']['map_balanced_success_delta_pp']['std']
            for name in pair_names]
    fig, axis = plt.subplots(figsize=(max(12, len(pair_names) * 1.2), 5), constrained_layout=True)
    axis.axhline(0, color='black', linewidth=1)
    axis.bar(np.arange(len(pair_names)), means, yerr=stds, capsize=4)
    axis.set_ylabel('map-balanced SR difference (percentage points)')
    axis.set_title('Paired val-unseen effect: mean ± seed standard deviation')
    axis.set_xticks(np.arange(len(pair_names)), pair_names, rotation=30, ha='right')
    fig.savefig(output_dir / 'multiseed_paired_deltas.png', dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME:SEED=/absolute/run/path')
    parser.add_argument('--pair', action='append', default=[], help='LEFT:RIGHT')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = {}
    loaded = {}
    for specification in args.run:
        identity, path = specification.split('=', 1)
        name, seed_text = identity.rsplit(':', 1)
        seed = int(seed_text)
        paths.setdefault(name, {})[str(seed)] = str(Path(path).resolve())
        loaded.setdefault(name, {})[seed] = load_run(Path(path))
    pairs = [tuple(specification.split(':', 1)) for specification in args.pair]
    result = aggregate(loaded, pairs)
    result['paths'] = paths
    (args.output_dir / 'summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    plot(result, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
