"""Evaluate exact-landmark, different-target instruction contrasts without GT inputs."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'multiagent')]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def landmark_key(objects, trajectory):
    obj = objects[trajectory.map_name][trajectory.object_id]
    description = obj.processed_descriptions[trajectory.desc_id]
    return tuple(sorted(name.strip() for name in description.landmarks))


def select_pairs(split, count, seed=20260911):
    from multiagent.cityreferobject import get_city_refer_objects
    from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
    objects = get_city_refer_objects()
    groups = defaultdict(list)
    for trajectory in load_mturk_trajectories(split, 'all', 50.0):
        key = landmark_key(objects, trajectory)
        if key:
            groups[(trajectory.map_name, key)].append(trajectory)
    rng = random.Random(seed)
    by_map = defaultdict(list)
    for (map_name, landmarks), entries in sorted(groups.items()):
        entries = sorted(entries, key=lambda item: (item.object_id, item.desc_id))
        rng.shuffle(entries)
        used = set()
        for index, left in enumerate(entries):
            left_id = (left.map_name, left.object_id, left.desc_id)
            if left_id in used:
                continue
            right = next((candidate for candidate in entries[index + 1:]
                          if candidate.object_id != left.object_id
                          and (candidate.map_name, candidate.object_id, candidate.desc_id) not in used
                          and candidate.target_position.xy.dist_to(left.target_position.xy) >= 20), None)
            if right is None:
                continue
            right_id = (right.map_name, right.object_id, right.desc_id)
            used.update((left_id, right_id))
            by_map[map_name].append((left, right, landmarks))
    for pairs in by_map.values():
        rng.shuffle(pairs)
    selected = []
    while len(selected) < count and any(by_map.values()):
        for map_name in sorted(by_map):
            if by_map[map_name]:
                selected.append(by_map[map_name].pop())
                if len(selected) == count:
                    break
    if len(selected) < count:
        raise ValueError(f'only {len(selected)} contrast pairs available')
    return objects, selected


def pair_manifest(objects, pairs):
    rows = []
    for left, right, landmarks in pairs:
        def details(item):
            obj = objects[item.map_name][item.object_id]
            return {'id': [item.map_name, item.object_id, item.desc_id],
                    'instruction': obj.descriptions[item.desc_id],
                    'target': list(item.target_position.xy)}
        rows.append({'map': left.map_name, 'landmarks': list(landmarks),
                     'target_distance_m': left.target_position.xy.dist_to(right.target_position.xy),
                     'left': details(left), 'right': details(right)})
    return rows


def make_args(checkpoint, output):
    from multiagent.parser import parse_args
    saved = sys.argv
    sys.argv = ['contrast', '--mode', 'eval', '--batch_size', '2',
                '--disable_task_interaction', '--enable_region_grounding',
                '--checkpoint', str(checkpoint), '--output_dir', str(output)]
    args = parse_args()
    sys.argv = saved
    return args


def final_distance(item):
    return float(item['trajectory'][-1].xy.dist_to(item['goal']))


def evaluate(out, checkpoint, objects, pairs):
    import multiagent.env as env_module
    from multiagent.agent import NavCMTAgent
    selected = [item for pair in pairs for item in pair[:2]]
    original_loader = env_module.load_mturk_trajectories
    env_module.load_mturk_trajectories = lambda *unused, **kwargs: selected
    args = make_args(checkpoint, out)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    agent = NavCMTAgent(args, allow_ngpus=False)
    agent.load(str(checkpoint))

    def rollout(overrides):
        env = env_module.CityNavBatch('val_unseen', args, batch_size=2, seed=args.seed)
        agent.env = env
        agent.instruction_overrides = overrides
        agent.test(DataLoader(env, batch_size=1), env_name='val_unseen', feedback='student')
        return dict(agent.get_results()), env

    correct, correct_env = rollout({})
    overrides = {}
    for left, right, _ in pairs:
        left_obj, right_obj = objects[left.map_name][left.object_id], objects[right.map_name][right.object_id]
        overrides[(left.map_name, left.object_id, left.desc_id)] = right_obj.descriptions[right.desc_id]
        overrides[(right.map_name, right.object_id, right.desc_id)] = left_obj.descriptions[left.desc_id]
    swapped, swapped_env = rollout(overrides)
    env_module.load_mturk_trajectories = original_loader
    torch.save(correct, out / 'correct_predictions.pt')
    torch.save(swapped, out / 'swapped_predictions.pt')
    correct_metrics, _ = correct_env.eval_metrics(correct)
    swapped_metrics, _ = swapped_env.eval_metrics(swapped)
    (out / 'correct_metrics.json').write_text(json.dumps(correct_metrics, indent=2) + '\n')
    (out / 'swapped_metrics.json').write_text(json.dumps(swapped_metrics, indent=2) + '\n')

    rows = []
    for episode_id in sorted(correct):
        normal, wrong = correct[episode_id], swapped[episode_id]
        normal_goal, wrong_goal = normal['all_pred_goal'][0], wrong['all_pred_goal'][0]
        target = normal['goal']
        rows.append({
            'id': str(episode_id),
            'first_prediction_shift_m': normal_goal.dist_to(wrong_goal),
            'first_prediction_correct_advantage_m': wrong_goal.dist_to(target) - normal_goal.dist_to(target),
            'final_distance_correct': final_distance(normal),
            'final_distance_swapped': final_distance(wrong),
            'region_prediction_changed': normal['region_prediction'][0] != wrong['region_prediction'][0],
        })
    advantages = np.array([row['first_prediction_correct_advantage_m'] for row in rows])
    shifts = np.array([row['first_prediction_shift_m'] for row in rows])
    result = {
        'episodes': len(rows),
        'pairs': len(pairs),
        'correct': correct_metrics,
        'swapped': swapped_metrics,
        'sr_delta_correct_minus_swapped_pp': correct_metrics['sr'] - swapped_metrics['sr'],
        'mean_first_prediction_shift_m': float(shifts.mean()),
        'mean_first_prediction_correct_advantage_m': float(advantages.mean()),
        'correct_first_prediction_better_percent': float(100 * (advantages > 0).mean()),
        'first_region_prediction_change_percent': float(100 * np.mean([row['region_prediction_changed'] for row in rows])),
        'rows': rows,
    }
    (out / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(advantages, bins=15, color='#2563eb')
    axes[0].axvline(0, color='black', linewidth=1)
    axes[0].set_title('Correct vs swapped instruction at t=0')
    axes[0].set_xlabel('correct prediction advantage (m; positive is better)')
    axes[1].scatter([row['final_distance_swapped'] for row in rows],
                    [row['final_distance_correct'] for row in rows], alpha=.7)
    limit = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([0, limit], [0, limit], '--', color='black', linewidth=1)
    axes[1].set_xlabel('swapped instruction final error (m)')
    axes[1].set_ylabel('correct instruction final error (m)')
    axes[1].set_title('Closed-loop contrast')
    fig.savefig(out / 'contrast.png', dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    objects, pairs = select_pairs('val_unseen', args.pairs, args.seed)
    manifest = {'split': 'val_unseen', 'seed': args.seed, 'selection':
                'exact same nonempty landmark list, distinct objects, target distance >=20m; balanced round-robin by map',
                'pairs': pair_manifest(objects, pairs)}
    if args.checkpoint:
        manifest['checkpoint'] = str(args.checkpoint.resolve())
        manifest['checkpoint_sha256'] = sha256(args.checkpoint)
    manifest['git_head'] = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    manifest['script_sha256'] = sha256(Path(__file__))
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    if args.manifest_only:
        print(json.dumps({'pairs': len(pairs), 'maps': sorted({p[0].map_name for p in pairs})}))
        return
    if not args.checkpoint:
        parser.error('--checkpoint is required unless --manifest-only is used')
    evaluate(args.output_dir, args.checkpoint, objects, pairs)


if __name__ == '__main__':
    main()
