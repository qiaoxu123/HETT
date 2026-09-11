"""Audit CityNav split identity and map leakage without loading any model."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path('/home/tenant2/Workspace/hett-experiments/01-teacher-fix')
sys.path[:0] = [str(ROOT), str(ROOT / 'multiagent')]
SPLITS = ('train_seen', 'val_seen', 'val_unseen', 'test_unseen')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def collect():
    from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
    result = {}
    for split in SPLITS:
        trajectories = load_mturk_trajectories(split, 'all', 50.0)
        keys = {(item.map_name, item.object_id, item.desc_id) for item in trajectories}
        starts = {(item.map_name, round(item.start_pose.x, 6),
                   round(item.start_pose.y, 6), round(item.start_pose.yaw, 6))
                  for item in trajectories}
        result[split] = {
            'episodes': len(trajectories),
            'unique_target_description_keys': len(keys),
            'unique_starts': len(starts),
            'maps': sorted({item.map_name for item in trajectories}),
            '_keys': keys, '_starts': starts,
        }
    return result


def audit(rows):
    pairs = {}
    names = list(rows)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            pairs[f'{left}:{right}'] = {
                'target_description_overlap': len(rows[left]['_keys'] & rows[right]['_keys']),
                'start_pose_overlap': len(rows[left]['_starts'] & rows[right]['_starts']),
                'map_overlap': sorted(set(rows[left]['maps']) & set(rows[right]['maps'])),
            }
    checks = {
        'all_episode_keys_unique_within_split': all(
            row['episodes'] == row['unique_target_description_keys'] for row in rows.values()),
        'all_start_poses_unique_within_split': all(
            row['episodes'] == row['unique_starts'] for row in rows.values()),
        'no_episode_key_overlap_across_splits': all(
            row['target_description_overlap'] == 0 for row in pairs.values()),
        'no_start_pose_overlap_across_splits': all(
            row['start_pose_overlap'] == 0 for row in pairs.values()),
        'val_unseen_maps_disjoint_from_seen': not (
            set(rows['val_unseen']['maps']) &
            (set(rows['train_seen']['maps']) | set(rows['val_seen']['maps']))),
        'test_unseen_maps_disjoint_from_train_and_validation': not (
            set(rows['test_unseen']['maps']) &
            (set(rows['train_seen']['maps']) | set(rows['val_seen']['maps']) |
             set(rows['val_unseen']['maps']))),
    }
    return checks, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = collect()
    checks, pairs = audit(rows)
    if not all(checks.values()):
        raise AssertionError(checks)
    public_rows = {split: {key: value for key, value in row.items()
                           if not key.startswith('_')}
                   for split, row in rows.items()}
    inputs = [ROOT / 'data/cityrefer/objects.json',
              ROOT / 'data/cityrefer/processed_descriptions.json']
    inputs += sorted((ROOT / 'data/processed_citynav').glob('citynav_*.json'))
    report = {
        'checks': checks,
        'splits': public_rows,
        'pairwise': pairs,
        'input_sha256': {str(path.resolve()): sha256(path) for path in inputs},
        'code_git_head': subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'interpretation': ('train_seen and val_seen intentionally share seen maps but not episodes, target '
                           'description keys, or starts. Both unseen splits use disjoint maps.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'checks': checks,
                      'episodes': {name: row['episodes'] for name, row in public_rows.items()},
                      'maps': {name: len(row['maps']) for name, row in public_rows.items()}},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
