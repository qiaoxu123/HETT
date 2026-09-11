"""Summarize and plot isolated HETT evaluation runs, including paired changes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def distance(pose, goal):
    return float(pose.xy.dist_to(goal))


def path_length(poses):
    return float(sum(a.xy.dist_to(b.xy) for a, b in zip(poses[:-1], poses[1:])))


def episode_rows(predictions, success_distance=20.0):
    rows = {}
    for episode_id, item in predictions.items():
        trajectory = item['trajectory']
        goal = item['goal']
        distances = np.array([distance(pose, goal) for pose in trajectory])
        final_success = bool(distances[-1] <= success_distance)
        any_success = bool((distances <= success_distance).any())
        gt_length = path_length(item.get('gt_trajectory', []))
        actual_length = path_length(trajectory)
        events = item.get('control_events', [])
        previous_events = ['coarse', *events[:-1]]
        switches = sum(previous != 'fine' and current == 'fine'
                       for previous, current in zip(previous_events, events))
        recoveries = sum(previous == 'fine' and current == 'coarse'
                         for previous, current in zip(previous_events, events))
        switch_distances = [distance(trajectory[min(index, len(trajectory) - 1)], goal)
                            for index, (previous, current) in enumerate(
                                zip(previous_events, events))
                            if previous != 'fine' and current == 'fine']
        stage1 = item.get('stage1_trajectory', [])
        stage2 = item.get('stage2_trajectory', [])
        if not events and stage2:
            switches = 1
            switch_distances = [distance(stage2[0], goal)]
        actions = max(0, len(trajectory) - 1)
        progress = item.get('progress', [])
        stopped = bool((events and events[-1] == 'stop') or len(progress) > actions)
        row = {
            'id': str(episode_id),
            'map': episode_id[0] if isinstance(episode_id, tuple) else str(episode_id).split(',')[0],
            'success': final_success,
            'any_success': any_success,
            'hit_then_lost': any_success and not final_success,
            'final_distance': float(distances[-1]),
            'best_distance': float(distances.min()),
            'actions': actions,
            'stage1_actions': max(0, len(stage1) - 1),
            'stage2_actions': max(0, len(stage2) - 1),
            'path_length': actual_length,
            'spl': float(final_success * gt_length / max(gt_length, actual_length, 1e-12)),
            'switches': int(switches),
            'recoveries': int(recoveries),
            'switch_distance_sum': float(sum(switch_distances)),
            'switch_distance_count': len(switch_distances),
            'stopped': stopped,
            'false_stop': stopped and not final_success,
            'stop_distance_sum': float(distances[-1]) if stopped else 0.0,
            'stop_count': int(stopped),
            'progress_at_stop_sum': float(progress[-1]) if stopped and progress else 0.0,
            'progress_at_stop_count': int(stopped and bool(progress)),
        }
        region_prediction = item.get('region_prediction', [])
        region_target = item.get('gt_region', [])
        if region_prediction and len(region_prediction) == len(region_target):
            pred = np.asarray(region_prediction)
            target = np.asarray(region_target)
            visible = target < 49
            row['region_correct'] = int((pred == target).sum())
            row['region_count'] = len(target)
            row['region_visible_correct'] = int(((pred == target) & visible).sum())
            row['region_visible_count'] = int(visible.sum())
            row['region_outside_correct'] = int(((pred == target) & ~visible).sum())
            row['region_outside_count'] = int((~visible).sum())
        hypothesis = item.get('hypothesis_indices', [])
        if hypothesis:
            maps = [indices[0] for indices in hypothesis]
            row['hypothesis_map_changes'] = sum(a != b for a, b in zip(maps[:-1], maps[1:]))
            row['hypothesis_transitions'] = max(0, len(maps) - 1)
            row['hypothesis_confidence_sum'] = float(sum(item.get('hypothesis_confidence', [])))
            row['hypothesis_steps'] = len(hypothesis)
        rows[str(episode_id)] = row
    return rows


def ratio(rows, numerator, denominator=None):
    if denominator is None:
        values = [float(row[numerator]) for row in rows.values()]
        return float(np.mean(values)) if values else None
    top = sum(row.get(numerator, 0) for row in rows.values())
    bottom = sum(row.get(denominator, 0) for row in rows.values())
    return float(top / bottom) if bottom else None


def percent_ratio(rows, numerator, denominator=None):
    value = ratio(rows, numerator, denominator)
    return None if value is None else 100 * value


def summarize_rows(rows):
    return {
        'episodes': len(rows),
        'success_percent': 100 * ratio(rows, 'success'),
        'oracle_success_percent': 100 * ratio(rows, 'any_success'),
        'hit_then_lost_percent': 100 * ratio(rows, 'hit_then_lost'),
        'mean_final_distance': ratio(rows, 'final_distance'),
        'mean_best_distance': ratio(rows, 'best_distance'),
        'mean_actions': ratio(rows, 'actions'),
        'mean_stage1_actions': ratio(rows, 'stage1_actions'),
        'mean_stage2_actions': ratio(rows, 'stage2_actions'),
        'mean_spl_percent': 100 * ratio(rows, 'spl'),
        'mean_switches': ratio(rows, 'switches'),
        'mean_recoveries': ratio(rows, 'recoveries'),
        'mean_switch_distance_m': ratio(rows, 'switch_distance_sum', 'switch_distance_count'),
        'stopped_percent': percent_ratio(rows, 'stopped'),
        'false_stop_percent_of_episodes': percent_ratio(rows, 'false_stop'),
        'false_stop_percent_of_stops': percent_ratio(rows, 'false_stop', 'stop_count'),
        'mean_stop_distance_m': ratio(rows, 'stop_distance_sum', 'stop_count'),
        'mean_predicted_progress_at_stop': ratio(
            rows, 'progress_at_stop_sum', 'progress_at_stop_count'),
        'region_accuracy_percent': (percent_ratio(rows, 'region_correct', 'region_count')
                                    if any('region_count' in row for row in rows.values()) else None),
        'region_visible_accuracy_percent': (percent_ratio(
                                            rows, 'region_visible_correct', 'region_visible_count')
                                            if any('region_count' in row for row in rows.values()) else None),
        'region_outside_accuracy_percent': (percent_ratio(
                                            rows, 'region_outside_correct', 'region_outside_count')
                                            if any('region_count' in row for row in rows.values()) else None),
        'hypothesis_map_change_percent': (percent_ratio(
                                          rows, 'hypothesis_map_changes', 'hypothesis_transitions')
                                          if any('hypothesis_steps' in row for row in rows.values()) else None),
        'mean_hypothesis_confidence': (ratio(rows, 'hypothesis_confidence_sum', 'hypothesis_steps')
                                       if any('hypothesis_steps' in row for row in rows.values()) else None),
    }


def paired_summary(left, right, bootstrap_samples=5000, seed=20260911):
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError('paired runs have no common episode ids')
    deltas = np.array([float(right[key]['success']) - float(left[key]['success']) for key in common])
    distance_deltas = np.array([right[key]['final_distance'] - left[key]['final_distance'] for key in common])
    by_map = defaultdict(list)
    for key, delta in zip(common, deltas):
        by_map[left[key]['map']].append(delta)
    map_means = np.array([np.mean(by_map[name]) for name in sorted(by_map)])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(map_means, size=(bootstrap_samples, len(map_means)), replace=True).mean(axis=1)
    return {
        'episodes': len(common),
        'maps': len(map_means),
        'success_delta_pp': float(100 * deltas.mean()),
        'success_wins': int((deltas > 0).sum()),
        'success_losses': int((deltas < 0).sum()),
        'mean_final_distance_delta_m': float(distance_deltas.mean()),
        'map_balanced_success_delta_pp': float(100 * map_means.mean()),
        'map_bootstrap_95ci_pp': [float(100 * x) for x in np.quantile(sampled, [.025, .975])],
    }


def load_run(path):
    result = {}
    for prediction_path in sorted((path / 'evaluation').glob('*_predictions.pt')):
        split = prediction_path.name.removesuffix('_predictions.pt')
        predictions = torch.load(prediction_path, map_location='cpu', weights_only=False)
        rows = episode_rows(predictions)
        metric_path = path / 'evaluation' / f'{split}_metrics.json'
        result[split] = {'reported': json.loads(metric_path.read_text()),
                         'derived': summarize_rows(rows), 'episodes': rows}
    if not result:
        raise FileNotFoundError(f'no evaluation predictions under {path}')
    return result


def plot_summary(runs, output):
    splits = ['val_unseen', 'test_unseen']
    labels, sr, spl, ne = [], [], [], []
    for name, run in runs.items():
        for split in splits:
            if split in run:
                labels.append(f'{name}\n{split}')
                reported = run[split]['reported']
                sr.append(reported['sr'])
                spl.append(reported['spl'])
                ne.append(reported['ne'])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.2), 4.5), constrained_layout=True)
    axes[0].bar(x - .18, sr, .36, label='SR')
    axes[0].bar(x + .18, spl, .36, label='SPL')
    axes[0].set_ylabel('percent')
    axes[0].set_xticks(x, labels, rotation=25, ha='right')
    axes[0].legend()
    axes[0].set_title('Navigation success')
    axes[1].bar(x, ne, color='#d97706')
    axes[1].set_ylabel('meters (lower is better)')
    axes[1].set_xticks(x, labels, rotation=25, ha='right')
    axes[1].set_title('Navigation error')
    fig.savefig(output, dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME=/absolute/run/path')
    parser.add_argument('--pair', action='append', default=[], help='LEFT:RIGHT')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    paths = {}
    for specification in args.run:
        name, path = specification.split('=', 1)
        paths[name] = str(Path(path).resolve())
        runs[name] = load_run(Path(path))
    pairs = {}
    for specification in args.pair:
        left, right = specification.split(':', 1)
        pairs[specification] = {}
        for split in sorted(set(runs[left]) & set(runs[right])):
            pairs[specification][split] = paired_summary(
                runs[left][split]['episodes'], runs[right][split]['episodes'])
    serializable = {'paths': paths, 'runs': {}, 'pairs': pairs}
    for name, run in runs.items():
        serializable['runs'][name] = {
            split: {'reported': values['reported'], 'derived': values['derived']}
            for split, values in run.items()
        }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + '\n')
    plot_summary(runs, args.output_dir / 'comparison.png')
    print(json.dumps(serializable, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
