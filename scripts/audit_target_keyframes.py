"""Audit target-only view selection without loading RGB or a GPU model."""

import argparse
import json

import numpy as np

from multiagent.cityreferobject import get_city_refer_objects
from multiagent.dataset.generate import generate_episodes_from_mturk_trajectories
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.target_only import pose_goal_distance, select_keyframe_pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--altitude', type=float, default=50.0)
    parser.add_argument('--near-min-m', type=float, default=25.0)
    parser.add_argument('--near-max-m', type=float, default=60.0)
    parser.add_argument('--mode', choices=['final', 'near_ring', 'closest', 'start'],
                        default='final')
    parser.add_argument('--splits', nargs='+', default=['train_seen', 'val_seen', 'val_unseen'])
    args = parser.parse_args()
    objects = get_city_refer_objects()
    output = {}
    for split in args.splits:
        episodes = generate_episodes_from_mturk_trajectories(
            objects, load_mturk_trajectories(split, 'all', args.altitude)
        )
        distances = []
        exact_ring = 0
        for episode in episodes:
            all_distances = np.asarray([
                pose_goal_distance(pose, episode.target_position)
                for pose in episode.trajectory
            ])
            exact_ring += int(np.any(
                (all_distances >= args.near_min_m)
                & (all_distances <= args.near_max_m)
            ))
            pose = select_keyframe_pose(
                episode.trajectory, episode.target_position, args.mode,
                args.near_min_m, args.near_max_m,
            )
            distances.append(pose_goal_distance(pose, episode.target_position))
        distances = np.asarray(distances)
        output[split] = {
            'mode': args.mode,
            'episodes': len(episodes),
            'exact_ring_available_pct': exact_ring / max(len(episodes), 1) * 100.0,
            'selected_distance_mean_m': float(distances.mean()),
            'selected_distance_median_m': float(np.median(distances)),
            'selected_inside_20_pct': float((distances <= 20).mean() * 100.0),
            'selected_inside_25_pct': float((distances < args.near_min_m).mean() * 100.0),
            'selected_over_60_pct': float((distances > args.near_max_m).mean() * 100.0),
        }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
