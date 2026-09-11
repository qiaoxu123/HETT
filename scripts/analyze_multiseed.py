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


CASE_FIELDS = (
    'id', 'map', 'success', 'any_success', 'hit_then_lost', 'false_stop',
    'stopped', 'final_distance', 'best_distance', 'actions', 'stage1_actions',
    'stage2_actions', 'switches', 'recoveries',
)


def compact_case(row):
    return {name: row.get(name) for name in CASE_FIELDS}


def select_failure_cases(loaded, pairs, limit=20):
    """Build readable indices into the preserved per-episode trajectories."""
    output = {'runs': {}, 'pairs': {}}
    for name, seeds in loaded.items():
        output['runs'][name] = {}
        for seed, run in sorted(seeds.items()):
            output['runs'][name][str(seed)] = {}
            for split, values in run.items():
                rows = list(values['episodes'].values())
                worst = sorted(rows, key=lambda row: row['final_distance'], reverse=True)[:limit]
                hit_lost = sorted((row for row in rows if row.get('hit_then_lost')),
                                  key=lambda row: row['final_distance'], reverse=True)[:limit]
                false_stop = sorted((row for row in rows if row.get('false_stop')),
                                    key=lambda row: row['final_distance'], reverse=True)[:limit]
                output['runs'][name][str(seed)][split] = {
                    'episodes': len(rows),
                    'worst_final_distance': [compact_case(row) for row in worst],
                    'hit_then_lost': [compact_case(row) for row in hit_lost],
                    'false_stops': [compact_case(row) for row in false_stop],
                }
    for left, right in pairs:
        identity = f'{left}:{right}'
        output['pairs'][identity] = {}
        for seed in sorted(set(loaded[left]) & set(loaded[right])):
            output['pairs'][identity][str(seed)] = {}
            splits = sorted(set(loaded[left][seed]) & set(loaded[right][seed]))
            for split in splits:
                before = loaded[left][seed][split]['episodes']
                after = loaded[right][seed][split]['episodes']
                common = sorted(set(before) & set(after))

                def paired_row(key):
                    old, new = before[key], after[key]
                    return {
                        'id': key, 'map': old['map'],
                        'before_success': old['success'], 'after_success': new['success'],
                        'before_final_distance': old['final_distance'],
                        'after_final_distance': new['final_distance'],
                        'final_distance_delta': new['final_distance'] - old['final_distance'],
                    }

                cases = [paired_row(key) for key in common]
                regressions = sorted((row for row in cases if row['before_success'] and
                                      not row['after_success']),
                                     key=lambda row: row['final_distance_delta'], reverse=True)
                rescues = sorted((row for row in cases if not row['before_success'] and
                                  row['after_success']),
                                 key=lambda row: row['final_distance_delta'])
                distance_regressions = sorted(
                    (row for row in cases if row['final_distance_delta'] > 0),
                    key=lambda row: row['final_distance_delta'], reverse=True)
                output['pairs'][identity][str(seed)][split] = {
                    'success_regressions': regressions[:limit],
                    'success_rescues': rescues[:limit],
                    'largest_distance_regressions': distance_regressions[:limit],
                }
    return output


def plot_failure_rates(loaded, output):
    labels, hit_lost, false_stop = [], [], []
    for name, seeds in loaded.items():
        for seed, run in sorted(seeds.items()):
            for split, values in run.items():
                rows = list(values['episodes'].values())
                labels.append(f'{name}:s{seed}\n{split}')
                hit_lost.append(100 * np.mean([bool(row.get('hit_then_lost')) for row in rows]))
                false_stop.append(100 * np.mean([bool(row.get('false_stop')) for row in rows]))
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(labels) * .8), 8),
                             constrained_layout=True)
    for axis, values, title, color in (
            (axes[0], hit_lost, 'Reached success radius, then finished outside', '#d97706'),
            (axes[1], false_stop, 'Stopped outside success radius', '#dc2626')):
        axis.bar(x, values, color=color)
        axis.set_ylabel('episodes (%)')
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=35, ha='right')
        axis.grid(axis='y', alpha=.15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def metric_text(metric, digits=2):
    low, high = metric['ci95']
    return (f"{metric['mean']:.{digits}f} ± {metric['std']:.{digits}f} "
            f"(95% CI {low:.{digits}f}…{high:.{digits}f})")


def render_report(result):
    lines = [
        '# HETT 三种子导航结果', '',
        '这里的均值和标准差按训练 seed 计算；95% 区间使用小样本 t 区间。',
        '导航效果以闭环 SR/SPL/NE 为准，loss 大小不等于任务贡献。', '',
        '## 各方案', '',
        '| 方案 | split | SR (%) | SPL (%) | NE (m) |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for name, splits in result['runs'].items():
        for split, metrics in splits.items():
            if not all(key in metrics for key in ('sr', 'spl', 'ne')):
                continue
            lines.append(
                f"| {name} | {split} | {metric_text(metrics['sr'])} | "
                f"{metric_text(metrics['spl'])} | {metric_text(metrics['ne'])} |")
    lines.extend(['', '## 配对变化', '',
                  '正的 SR 变化代表右侧方案更好；NE 变化为负代表终点更近。', '',
                  '| 对比（左→右） | split | SR 变化 (pp) | 地图平衡 SR 变化 (pp) | NE 变化 (m) | 救回/退化 |',
                  '| --- | --- | ---: | ---: | ---: | ---: |'])
    for name, splits in result['pairs'].items():
        for split, values in splits.items():
            lines.append(
                f"| {name} | {split} | {metric_text(values['success_delta_pp'])} | "
                f"{metric_text(values['map_balanced_success_delta_pp'])} | "
                f"{metric_text(values['mean_final_distance_delta_m'])} | "
                f"{values['success_wins_total']}/{values['success_losses_total']} |")
    lines.extend([
        '', '## 配套文件', '',
        '- `summary.json`：完整统计与逐 seed 数值。',
        '- `failure_cases.json`：最差终点、错误停止、到达后丢失以及配对救回/退化案例索引。',
        '- `multiseed_metrics.png`、`multiseed_paired_deltas.png`、`failure_cases.png`：结果图。',
        '', '注意：只有冻结流程生成的 test-unseen 表才能作为最终测试结论；开发阶段的 val-unseen 不应被写成 test 结果。',
        '',
    ])
    return '\n'.join(lines)


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
    failures = select_failure_cases(loaded, pairs)
    (args.output_dir / 'failure_cases.json').write_text(
        json.dumps(failures, indent=2, ensure_ascii=False) + '\n')
    plot_failure_rates(loaded, args.output_dir / 'failure_cases.png')
    (args.output_dir / 'REPORT.md').write_text(render_report(result))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
