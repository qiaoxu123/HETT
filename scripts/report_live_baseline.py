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


def stage_diagnostics(validation):
    result = {}
    for split, metrics in validation.items():
        total_length = float(metrics['lengths'])
        result[split] = {
            'coarse_endpoint_ne_m': float(metrics['stage1_ne']),
            'final_ne_m': float(metrics['ne']),
            'fine_refinement_ne_change_m': float(metrics['ne'] - metrics['stage1_ne']),
            'coarse_sr_percent': float(metrics['sr1']),
            'final_sr_percent': float(metrics['sr']),
            'fine_refinement_sr_change_pp': float(metrics['sr'] - metrics['sr1']),
            'oracle_to_final_sr_gap_pp': float(metrics['oracle_sr'] - metrics['sr']),
            'stage2_path_share_percent': (100 * float(metrics['stage2_length']) / total_length
                                          if total_length else None),
            'interpretation': ('coarse endpoint NE is the available aggregate proxy for true '
                               'switch distance; positive refinement NE change means final '
                               'navigation ended farther from the real goal'),
        }
    return result


def render_markdown(report):
    rows = report['completed_epochs']
    lines = [
        '# 原版结构基线：实时训练报告', '',
        '轮次按已完成数量从 1 起。SR/SPL 越高越好，NE 越低越好。',
        'loss 占比只描述数值尺度，不代表因果重要性。', '',
        '| 完成轮次 | IL loss | val-seen SR/SPL/NE | val-unseen SR/SPL/NE |',
        '| ---: | ---: | ---: | ---: |',
    ]
    for item in rows:
        seen = item['validation']['val_seen']
        unseen = item['validation']['val_unseen']
        lines.append(
            f"| {item['epoch']} | {item['loss']['reconstructed_il_loss']:.4f} | "
            f"{seen['sr']:.2f} / {seen['spl']:.2f} / {seen['ne']:.2f}m | "
            f"{unseen['sr']:.2f} / {unseen['spl']:.2f} / {unseen['ne']:.2f}m |")
    if not rows:
        lines.extend(['', '尚无完整轮次。', ''])
        return '\n'.join(lines)
    latest = rows[-1]
    lines.extend(['', f"## 最新完整轮次：{latest['epoch']}", ''])
    if len(rows) > 1:
        previous = rows[-2]
        lines.extend(['相对上一轮：', '',
                      f"- IL loss：{latest['loss']['reconstructed_il_loss'] - previous['loss']['reconstructed_il_loss']:+.4f}"])
        for split in ('val_seen', 'val_unseen'):
            old, new = previous['validation'][split], latest['validation'][split]
            lines.append(
                f"- {split}：SR {new['sr'] - old['sr']:+.2f}pp，"
                f"SPL {new['spl'] - old['spl']:+.2f}pp，NE {new['ne'] - old['ne']:+.2f}m。")
    lines.extend(['', 'fine refinement 对粗阶段终点的变化：', ''])
    for split in ('val_seen', 'val_unseen'):
        values = latest['stage_diagnostics'][split]
        lines.append(
            f"- {split}：SR {values['fine_refinement_sr_change_pp']:+.2f}pp，"
            f"NE {values['fine_refinement_ne_change_m']:+.2f}m，"
            f"stage2 路径占比 {values['stage2_path_share_percent']:.2f}%。")
    shares = latest['loss']['share_percent']
    lines.extend(['', '最新加权 loss 数值占比：', '',
                  '- ' + '，'.join(f'{name} {shares[name]:.2f}%'
                                  for name in ('direction', 'progress', 'goal', 'target')) + '。'])
    health = report['gradient_health'].get(str(latest['epoch']), {})
    if health:
        lines.extend(['',
                      f"梯度采样：p95={health['grad_p95']:.2f}，最大={health['grad_max']:.2f}，"
                      f"全部有限={health['all_finite']}。日志是 ET 主体裁剪前范数，实际阈值为 40。"])
    lines.extend(['', '## 解读边界', '',
                  '- 单轮变化不是统计显著性结论，最终以完整 20 轮、三个 seed 和冻结后的 test 为准。',
                  '- fine 阶段可能提高边界样本成功率，同时拉长路径或恶化平均终点；必须同时看 SR、SPL、NE。',
                  ''])
    return '\n'.join(lines)


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
                'loss': loss_breakdown(row), 'validation': row['validation'],
                'stage_diagnostics': stage_diagnostics(row['validation'])}
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
    (args.output_dir / 'REPORT.md').write_text(render_markdown(report))

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

    if completed:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        x = np.array([item['epoch'] for item in completed])
        for split, color in [('val_seen', '#2563eb'), ('val_unseen', '#d97706')]:
            diagnostics = [item['stage_diagnostics'][split] for item in completed]
            axes[0].plot(x, [item['coarse_endpoint_ne_m'] for item in diagnostics],
                         marker='o', color=color, linestyle='--', label=f'{split} coarse end')
            axes[0].plot(x, [item['final_ne_m'] for item in diagnostics],
                         marker='s', color=color, label=f'{split} final')
            axes[1].plot(x, [item['coarse_sr_percent'] for item in diagnostics],
                         marker='o', color=color, linestyle='--', label=f'{split} coarse SR')
            axes[1].plot(x, [item['final_sr_percent'] for item in diagnostics],
                         marker='s', color=color, label=f'{split} final SR')
        axes[0].set(title='Does fine refinement reduce true-goal error?', xlabel='epoch',
                    ylabel='meters (lower is better)')
        axes[1].set(title='Does fine refinement recover success?', xlabel='epoch',
                    ylabel='percent (higher is better)')
        for axis in axes:
            axis.grid(alpha=.15)
            axis.legend(fontsize=8)
        fig.savefig(args.output_dir / 'stage_diagnostics.png', dpi=180)
        plt.close(fig)
    print(json.dumps({'completed_epochs': len(completed),
                      'batch_samples': len(batches),
                      'output': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
