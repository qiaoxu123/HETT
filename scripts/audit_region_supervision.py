"""Audit 7x7 target-region labels on previously saved student rollout states."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import cv2
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiagent.observation.cropclient import project_colrow_to_crop, region_target_from_crop
from multiagent.mapdata import GROUND_LEVEL
from multiagent.defaultpaths import ORTHO_IMAGE_DIR
from multiagent.space import view_area_corners
from multiagent.space import Point2D, Pose4D


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sampled-only', action='store_true',
                        help='audit probe sample states instead of every saved rollout state')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {'source': str(args.probe_dir.resolve()), 'splits': {}}
    split_counts = {}
    rasters = {}
    for split in ('train_seen', 'val_unseen'):
        manifest_path = args.probe_dir / f'{split}_manifest.json'
        labels_path = args.probe_dir / f'{split}_labels.npy'
        manifest = json.loads(manifest_path.read_text())
        labels = np.load(labels_path)
        samples = manifest['samples']
        if len(samples) != len(labels):
            raise ValueError(f'{split}: manifest/label length mismatch')
        if args.sampled_only:
            records = []
            for sample, delta in zip(samples, labels):
                pose = Pose4D(*sample['pose'])
                target = Point2D(pose.x + float(delta[0]), pose.y - float(delta[1]))
                records.append((sample['map'], pose, target))
            trajectory_path = None
            episodes = len({sample['id'] for sample in samples})
        else:
            trajectory_path = args.probe_dir / f'{split}_trajectories.pt'
            trajectories = torch.load(trajectory_path, map_location='cpu', weights_only=False)
            map_by_id = {sample['id']: sample['map'] for sample in samples}
            records = []
            for episode_id, trajectory in trajectories.items():
                map_name = map_by_id[str(episode_id)]
                records.extend((map_name, pose, trajectory['goal'])
                               for pose in trajectory['trajectory'])
            episodes = len(trajectories)
        counts = np.zeros(50, dtype=np.int64)
        raster_disagreements = 0
        max_projection_delta = 0.0
        for map_name, pose, target in records:
            world_corners = np.asarray(
                view_area_corners(pose, GROUND_LEVEL[map_name]), dtype=np.float32
            )
            pixel, visible = project_colrow_to_crop(
                world_corners, (target.x, target.y), (224, 224)
            )
            direct_label = region_target_from_crop(pixel, visible, (224, 224))
            if map_name not in rasters:
                rasters[map_name] = rasterio.open(ORTHO_IMAGE_DIR / f"{map_name}.tif")
            raster = rasters[map_name]
            corners = np.array([raster.index(x, y) for x, y in
                                view_area_corners(pose, GROUND_LEVEL[map_name])], dtype=np.float32)
            corners = np.flip(corners, axis=-1)
            row, col = raster.index(target.x, target.y)
            image_corners = np.array([(0, 0), (223, 0), (223, 223), (0, 223)], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(corners, image_corners)
            raster_pixel = cv2.perspectiveTransform(
                np.array([[[col, row]]], dtype=np.float32), transform
            ).reshape(2)
            raster_visible = bool((raster_pixel >= 0).all() and (raster_pixel <= 223).all())
            raster_label = region_target_from_crop(raster_pixel, raster_visible, (224, 224))
            counts[raster_label] += 1
            raster_disagreements += int(direct_label != raster_label)
            max_projection_delta = max(max_projection_delta, float(np.abs(pixel-raster_pixel).max()))
        visible = int(counts[:49].sum())
        split_counts[split] = counts
        report['splits'][split] = {
            'states': len(records),
            'episodes': episodes,
            'state_source': 'probe samples' if args.sampled_only else 'all saved student rollout states',
            'maps': manifest['maps'],
            'visible': visible,
            'outside': int(counts[49]),
            'visible_percent': 100.0 * visible / len(records),
            'region_counts': counts.tolist(),
            'raster_label_disagreements': raster_disagreements,
            'max_direct_vs_raster_pixel_delta': max_projection_delta,
            'manifest_sha256': sha256(manifest_path),
            'labels_sha256': sha256(labels_path),
            'trajectories_sha256': sha256(trajectory_path) if trajectory_path else None,
        }

    for raster in rasters.values():
        raster.close()

    output_json = args.output_dir / 'region_supervision_audit.json'
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, split in zip(axes, ('train_seen', 'val_unseen')):
        grid = split_counts[split][:49].reshape(7, 7)
        image = axis.imshow(grid, cmap='Blues')
        axis.set_title(f"{split}\nvisible={report['splits'][split]['visible_percent']:.1f}%")
        axis.set_xlabel('image column')
        axis.set_ylabel('image row')
        fig.colorbar(image, ax=axis, fraction=.046)
    fig.savefig(args.output_dir / 'region_supervision_distribution.png', dpi=180)
    print(json.dumps(report['splits'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
