"""Paired goal-point metrics for the frozen-encoder goal-head screen."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def distances(trajectory):
    goal = trajectory['goal']
    return [point.dist_to(goal) for point in trajectory.get('pred_goal', [])]


def summarize(rows):
    result = {'episodes': len(rows)}
    for name, selector in [('first', lambda x: x[0]), ('last', lambda x: x[-1]),
                           ('best', lambda x: min(x))]:
        values = np.array([selector(ds) if (ds := distances(row)) else np.inf
                           for row in rows.values()], dtype=float)
        finite = values[np.isfinite(values)]
        result[name] = {
            'hit20': 100 * float(np.mean(values <= 20)),
            'median_m': float(np.median(finite)) if len(finite) else None,
            'mean_m': float(np.mean(finite)) if len(finite) else None,
            'missing': int(np.sum(~np.isfinite(values))),
        }
    return result


def paired(parent, candidate):
    keys = sorted(set(parent) & set(candidate))
    output = {'episodes': len(keys)}
    for name, selector in [('first', lambda x: x[0]), ('last', lambda x: x[-1])]:
        before, after = [], []
        for key in keys:
            old, new = distances(parent[key]), distances(candidate[key])
            before.append(selector(old) if old else np.inf)
            after.append(selector(new) if new else np.inf)
        before, after = np.asarray(before), np.asarray(after)
        output[name] = {
            'parent_hit20': 100 * float(np.mean(before <= 20)),
            'candidate_hit20': 100 * float(np.mean(after <= 20)),
            'delta_pp': 100 * float(np.mean(after <= 20) - np.mean(before <= 20)),
            'improved_count': int(np.sum(after < before)),
            'worsened_count': int(np.sum(after > before)),
            'median_error_change_m': float(np.median(after - before)),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-eval', required=True, type=Path)
    parser.add_argument('--candidate-eval', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = {'splits': {}}
    for split in ('val_seen', 'val_unseen'):
        parent = load(args.parent_eval / f'{split}_predictions.pt')
        candidate = load(args.candidate_eval / f'{split}_predictions.pt')
        shared = {key: parent[key] for key in candidate if key in parent}
        report['splits'][split] = {
            'parent': summarize(shared), 'candidate': summarize(candidate),
            'paired': paired(parent, candidate),
            'navigation': {
                'parent': json.loads((args.parent_eval / f'{split}_metrics.json').read_text()),
                'candidate': json.loads((args.candidate_eval / f'{split}_metrics.json').read_text()),
            },
        }
    unseen = report['splits']['val_unseen']['paired']['last']['delta_pp']
    seen = report['splits']['val_seen']['paired']['last']['delta_pp']
    report['gate'] = {'unseen_last_hit20_delta_pp': unseen,
                      'seen_last_hit20_delta_pp': seen,
                      'pass': unseen >= 3 and seen >= -3}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = ['# 第一阶段坐标头筛选', '',
             '| 划分 | 父模型最后目标 Hit@20 | 微调后 | 变化 | 第一目标变化 |',
             '|---|---:|---:|---:|---:|']
    for split, row in report['splits'].items():
        last, first = row['paired']['last'], row['paired']['first']
        lines.append(f"| {split} | {last['parent_hit20']:.2f}% | {last['candidate_hit20']:.2f}% | "
                     f"{last['delta_pp']:+.2f} pp | {first['delta_pp']:+.2f} pp |")
    lines += ['', ('**通过筛选门槛。**' if report['gate']['pass'] else '**未通过筛选门槛。**'),
              '', '这是 512-episode、单 seed 的机制筛选，不是最终导航成绩。']
    (args.out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report['gate'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
