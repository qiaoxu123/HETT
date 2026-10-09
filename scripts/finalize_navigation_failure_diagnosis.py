#!/usr/bin/env python3
"""Add path metrics, flat CSVs and compact failure cases to a closed-loop run."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def dtw(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    acc = np.full((len(a) + 1, len(b) + 1), np.inf)
    acc[0, 0] = 0.
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            acc[i, j] = d[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[-1, -1])


def enrich(ep, radius=20.):
    path = np.asarray(ep['path_xy'], dtype=float)
    gt = np.asarray(ep['teacher_xy'], dtype=float)
    goal = np.asarray(ep['goal_xy'], dtype=float)
    dist = np.linalg.norm(path - goal, axis=-1)
    first = np.flatnonzero(dist <= radius)
    vec = np.diff(path, axis=0)
    lengths = np.linalg.norm(vec, axis=-1)
    moving = lengths > 1e-4
    turns = sum(abs(float(s.get('macro_yaw_delta_rad') or 0.)) > math.pi / 12
                for s in ep.get('steps', []))
    folds = 0
    compares = 0
    for u, v in zip(vec[:-1], vec[1:]):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu > 1e-4 and nv > 1e-4:
            compares += 1
            folds += float(np.dot(u, v) / (nu * nv) < -0.5)
    score = math.exp(-dtw(gt, path) / (radius * max(1, len(gt)))) if len(gt) and len(path) else 0.
    steps = ep.get('steps', [])
    selected_hits = [float(np.linalg.norm(np.asarray(s['endpoint_xy'], dtype=float) - goal) <= radius)
                     for s in steps if s.get('endpoint_xy') is not None]
    return {
        'first_enter_step': int(first[0]) if len(first) else None,
        'initial_in_success': bool(dist[0] <= radius),
        'ever_entered': bool(len(first)),
        'left_after_entry': bool(len(first) and dist[-1] > radius),
        'min_goal_distance_m': float(dist.min()),
        'final_goal_distance_m': float(dist[-1]),
        'mean_flight_step_m': float(lengths.mean()) if len(lengths) else 0.,
        'min_flight_step_m': float(lengths.min()) if len(lengths) else 0.,
        'zero_displacement_fraction': float(np.mean(~moving)) if len(moving) else 0.,
        'turn_count_gt15deg': int(turns),
        'trajectory_foldback_rate': folds / compares if compares else 0.,
        'selected_goal_hit20_rate': float(np.mean(selected_hits)) if selected_hits else None,
        'nDTW': score,
        'sDTW': score * float(ep['success']),
        'entered_then_left_rate': float(bool(len(first) and dist[-1] > radius)),
        'selected_goal_ids': [s.get('goal_id') for s in steps],
        'top20_hit_steps': [bool(s.get('trajectory_pool_hit')) for s in steps],
        'selected_hit_steps': [bool(np.linalg.norm(np.asarray(s['endpoint_xy'], dtype=float) - goal) <= radius)
                               if s.get('endpoint_xy') is not None else None for s in steps],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    opt = ap.parse_args()
    root = opt.run_dir
    summaries, episode_rows, current_failures = [], [], []
    for path in sorted((root / 'epoch01').glob('*.json')):
        obj = json.loads(path.read_text())
        s = dict(obj['summary'])
        eps = obj['episodes']
        rich = []
        for ep in eps:
            metrics = enrich(ep)
            rich_ep = dict(ep, **metrics)
            rich.append(rich_ep)
            episode_rows.append(dict(split=s['split'], variant=s['variant'],
                                     episode_id='|'.join(map(str, ep['episode_id'])),
                                     success=ep['success'], spl=ep['spl'], osr=ep['osr'],
                                     ne=ep['ne'], initial_distance_m=ep['initial_distance_m'],
                                     inference_seconds=ep.get('inference_seconds'),
                                     final_distance_m=metrics['final_goal_distance_m'],
                                     min_distance_m=metrics['min_goal_distance_m'],
                                     entered_then_left=metrics['left_after_entry'],
                                     selected_goal_hit20_rate=metrics['selected_goal_hit20_rate'],
                                     first_enter_step=metrics['first_enter_step'],
                                     nDTW=metrics['nDTW'], sDTW=metrics['sDTW'],
                                     mean_flight_step_m=metrics['mean_flight_step_m'],
                                     min_flight_step_m=metrics['min_flight_step_m'],
                                     zero_displacement_fraction=metrics['zero_displacement_fraction'],
                                     turns_gt15deg=metrics['turn_count_gt15deg'],
                                     foldback_rate=metrics['trajectory_foldback_rate'],
                                     goal_switches=sum(bool(st.get('goal_switch')) for st in ep.get('steps', [])),
                                     termination=ep['termination']))
        s.update(nDTW=float(np.mean([e['nDTW'] for e in rich])),
                 sDTW=float(np.mean([e['sDTW'] for e in rich])),
                 first_enter_step_mean=float(np.mean([e['first_enter_step'] for e in rich if e['first_enter_step'] is not None])) if any(e['first_enter_step'] is not None for e in rich) else None,
                 entered_then_left_rate=float(np.mean([e['left_after_entry'] for e in rich])),
                 actual_mean_flight_step_m=float(np.mean([e['mean_flight_step_m'] for e in rich])),
                 zero_displacement_fraction=float(np.mean([e['zero_displacement_fraction'] for e in rich])),
                 mean_turns_gt15deg=float(np.mean([e['turn_count_gt15deg'] for e in rich])),
                 mean_foldback_rate=float(np.mean([e['trajectory_foldback_rate'] for e in rich])))
        s.update(mean_final_distance_m=float(np.mean([e['final_goal_distance_m'] for e in rich])),
                 mean_min_distance_m=float(np.mean([e['min_goal_distance_m'] for e in rich])),
                 mean_selected_goal_hit20_rate=float(np.mean([e['selected_goal_hit20_rate'] for e in rich if e['selected_goal_hit20_rate'] is not None])) if any(e['selected_goal_hit20_rate'] is not None for e in rich) else None,
                 mean_turns_gt15deg=float(np.mean([e['turn_count_gt15deg'] for e in rich])),
                 mean_episode_flight_step_min_m=float(min((e['min_flight_step_m'] for e in rich), default=0.)),
                 mean_macro_zero_translation_rate=float(s.get('macro_zero_translation_rate') or 0.))
        latency = [float(e['inference_seconds']) for e in rich if e.get('inference_seconds') is not None]
        if latency:
            s.update(inference_seconds_min=float(np.min(latency)),
                     inference_seconds_mean=float(np.mean(latency)),
                     inference_seconds_variance=float(np.var(latency)),
                     inference_seconds_max=float(np.max(latency)))
        if s['variant'] in ('D', 'E'):
            current_failures.extend((not ep['left_after_entry'], ep['min_goal_distance_m'], s['split'], ep)
                                    for ep in rich if not ep['success'])
        s.pop('episodes_detail', None)
        obj['summary'] = s
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        summaries.append({k: v for k, v in s.items() if not k.startswith('episodes_detail')})
    if not summaries:
        raise SystemExit(f'No epoch01 JSON found under {root}')
    fields = sorted({k for row in summaries for k in row})
    with (root / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summaries)
    fields = sorted({k for row in episode_rows for k in row})
    with (root / 'per_episode.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(episode_rows)
    timed = [row for row in episode_rows if row.get('inference_seconds') is not None]
    timed.sort(key=lambda row: float(row['inference_seconds']), reverse=True)
    (root / 'slowest_episodes_top5.json').write_text(
        json.dumps(timed[:5], ensure_ascii=False, indent=2))
    # Ten illustrative current-controller failures, preferentially covering
    # successful-range entries that were subsequently lost.
    current_failures.sort(key=lambda x: (x[0], x[1]))
    cases = []
    seen_case_ids = set()
    for _, _, split, ep in current_failures:
        case_key = (split, tuple(ep['episode_id']))
        if case_key in seen_case_ids:
            continue
        seen_case_ids.add(case_key)
        steps = ep.get('steps', [])
        cases.append({
            'split': split, 'episode_id': ep['episode_id'],
            'start_xy': ep['path_xy'][0], 'target_xy': ep['goal_xy'],
            'selected_goal_last_xy': steps[-1].get('endpoint_xy') if steps else None,
            'final_xy': ep['path_xy'][-1], 'termination': ep['termination'],
            'success': ep['success'], 'initial_distance_m': ep['initial_distance_m'],
            'min_distance_m': ep['min_goal_distance_m'], 'final_distance_m': ep['final_goal_distance_m'],
            'first_enter_step': ep['first_enter_step'], 'goal_id_sequence': ep['selected_goal_ids'],
            'top20_hit_steps': ep['top20_hit_steps'], 'path_xy': ep['path_xy'],
            'stop_reasons': [ep['termination']],
        })
        if len(cases) >= 10:
            break
    (root / 'failure_cases_top10.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print(f'wrote {len(summaries)} summaries, {len(episode_rows)} episode rows, {len(cases)} failure cases')


if __name__ == '__main__':
    main()
