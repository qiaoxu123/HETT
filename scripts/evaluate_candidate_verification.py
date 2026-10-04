#!/usr/bin/env python3
"""Explicitly re-rank Static B0 peaks with language-matched visual attributes."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects  # noqa: E402
from multiagent.mapdata import GROUND_LEVEL, MAP_BOUNDS  # noqa: E402
from multiagent.models.multi_attribute_head import MultiAttributeHead  # noqa: E402
from multiagent.navigation_state import nms_topk_from_belief  # noqa: E402
from multiagent.observation import cropclient  # noqa: E402
from multiagent.space import Pose4D  # noqa: E402
from multiagent.static_belief_dataset import load_static_samples  # noqa: E402
from multiagent.visual_attributes.labels import geometry_features, shape_label, size_thresholds  # noqa: E402
from multiagent.visual_attributes.parser import parse_attributes  # noqa: E402

KS = (1, 4, 8, 16); RADII = (20.0, 40.0)


def required_attributes(instruction):
    parsed = parse_attributes(instruction); result = {}
    if parsed.get('color'): result['color'] = parsed['color'][0]
    sizes = parsed.get('size', ()); mapped_size = next(({'big': 'large'}.get(x, x) for x in sizes if x in ('small', 'large', 'big')), None)
    if mapped_size: result['size'] = mapped_size
    shapes = parsed.get('shape', ()); mapped_shape = next((x for x in shapes if x in ('square', 'rectangular', 'round')), None)
    if mapped_shape: result['shape'] = mapped_shape
    semantic = parsed.get('semantic', ())
    semantic_map = {'building': 'Building', 'car': 'Car', 'parking': 'Parking', 'parking_lot': 'Parking'}
    mapped_semantic = next((semantic_map[x] for x in semantic if x in semantic_map), None)
    if mapped_semantic: result['semantic'] = mapped_semantic
    context = set(parsed.get('context', ()))
    if context.intersection({'road', 'intersection', 'crossing'}): result['road_context'] = 'yes'
    if parsed.get('roof'): result['roof_presence'] = 'yes'
    return result


def candidate_world(map_name, row, col, grid=30, map_meters=410.0):
    bounds = MAP_BOUNDS[map_name]
    return bounds.x_min + (col + 0.5) * map_meters / grid, bounds.y_max - (row + 0.5) * map_meters / grid


def object_crop(map_name, xy, obj, altitude=60.0, size=256):
    pose = Pose4D(xy[0], xy[1], GROUND_LEVEL[map_name] + altitude, 0.0)
    rgb = cropclient.crop_image(map_name, pose, (size, size), 'rgb')
    polygon = []
    for point in obj.contour:
        image_row = (xy[0] + altitude - point.x) / (2 * altitude) * (size - 1)
        image_col = (xy[1] + altitude - point.y) / (2 * altitude) * (size - 1)
        polygon.append((round(image_col), round(image_row)))
    polygon = np.asarray(polygon, np.int32)
    x, y, width, height = cv2.boundingRect(polygon); padding = max(4, round(max(width, height) * 0.25))
    x0, y0 = max(0, x - padding), max(0, y - padding); x1, y1 = min(size, x + width + padding), min(size, y + height + padding)
    crop = rgb[y0:y1, x0:x1]
    return cv2.resize(crop if crop.size else rgb, (size, size), interpolation=cv2.INTER_AREA)


def metric(rows):
    result = {'samples': len(rows)}
    for key in ('top1_distance_m', 'oracle_distance@8_m', 'oracle_distance@16_m'):
        result[key] = float(np.mean([row[key] for row in rows]))
    for k in KS:
        for radius in RADII: result[f'recall@{k}/{int(radius)}m'] = float(np.mean([row[f'recall@{k}/{int(radius)}m'] for row in rows]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data'); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--b0-cache-root', required=True, type=Path); parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--model-name', default='google/siglip2-base-patch16-256'); parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--pool-k', type=int, default=16); parser.add_argument('--max-samples', type=int)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if 'test_unseen' in ' '.join(map(str, vars(args).values())): raise ValueError('test_unseen is forbidden')
    from transformers import AutoImageProcessor, AutoModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained(args.model_name); encoder = AutoModel.from_pretrained(args.model_name).to(device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False); encoder.load_state_dict(saved['model'])
    classes = saved['classes']; head = MultiAttributeHead(encoder.config.vision_config.hidden_size, {k: len(v) for k, v in classes.items()}).to(device)
    head.load_state_dict(saved['head']); encoder.eval(); head.eval()
    objects = get_city_refer_objects(args.data_root / 'cityrefer/objects.json', args.data_root / 'cityrefer/processed_descriptions.json')
    positions = {name: (list(values.values()), np.asarray([[obj.position.x, obj.position.y] for obj in values.values()])) for name, values in objects.items()}
    train_rows = json.loads((args.data_root / 'processed_citynav' / 'citynav_train_seen.json').read_text())
    train_objects = {f"{row['area']}_block_{row['block']}:{int(row['object_ids'][0])}" for row in train_rows}
    thresholds = size_thresholds([
        objects[map_name][int(object_id)] for map_name, object_id in (key.rsplit(':', 1) for key in train_objects)
    ])
    cropclient.load_image_cache(args.data_root / 'rgbd'); all_results = {}
    for split in ('val_seen', 'val_unseen'):
        samples = load_static_samples(args.data_root, split); beliefs = torch.load(args.b0_cache_root / f'{split}_b0_eall.pt', map_location='cpu', weights_only=True)
        if args.max_samples: samples, beliefs = samples[:args.max_samples], beliefs[:args.max_samples]
        records, pending_images, pending_meta = [], [], []; started = time.time()
        def flush_predictions():
            if not pending_images: return
            pixels = processor(images=list(pending_images), return_tensors='pt')['pixel_values'].to(device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                output = head(encoder.get_image_features(pixel_values=pixels))
            probabilities = {name: torch.softmax(logits.float(), -1).cpu().numpy() for name, logits in output.items()}
            for flat_index, (record_index, candidate_index) in enumerate(pending_meta):
                records[record_index]['candidates'][candidate_index]['visual'] = {
                    name: {label: float(probabilities[name][flat_index, index]) for index, label in enumerate(labels)}
                    for name, labels in classes.items()
                }
            pending_images.clear(); pending_meta.clear()
        for sample_index, (sample, belief) in enumerate(zip(samples, beliefs)):
            required = required_attributes(sample.instruction); candidates = nms_topk_from_belief(belief, top_k=args.pool_k, kernel_size=3)
            map_objects, object_xy = positions[sample.map_name]
            record = {'required': required, 'goal': sample.goal_row_col, 'candidates': []}; records.append(record)
            for candidate_index, candidate in enumerate(candidates):
                xy = candidate_world(sample.map_name, candidate.row, candidate.col)
                nearest_index = int(np.argmin(np.sum((object_xy - np.asarray(xy)) ** 2, axis=1))); obj = map_objects[nearest_index]
                nearest_distance = float(np.linalg.norm(object_xy[nearest_index] - np.asarray(xy)))
                geometry = geometry_features(obj)
                record['candidates'].append({'row': candidate.row, 'col': candidate.col, 'b0': candidate.probability,
                                             'nearest_distance_m': nearest_distance,
                                             'geometry': {'size': 'small' if geometry['area_m2'] < thresholds[0] else 'medium' if geometry['area_m2'] < thresholds[1] else 'large',
                                                          'shape': shape_label(geometry)}})
                pending_images.append(object_crop(sample.map_name, xy, obj)); pending_meta.append((sample_index, candidate_index))
                if len(pending_images) >= args.batch_size: flush_predictions()
        flush_predictions()
        # Tune one explicit weight on val_seen only; val_unseen uses that frozen choice.
        all_results[split] = {'records': records, 'inference_seconds': time.time() - started}

    weights = (0.0, 0.25, 0.5, 1.0, 2.0)
    variants = {'b0': (), 'color': ('color',), 'size': ('size',), 'shape': ('shape',), 'context': ('road_context', 'roof_presence'),
                'semantic': ('semantic',), 'all_visual': ('color', 'size', 'shape', 'road_context', 'roof_presence', 'semantic'),
                'geometry': ('geometry:size', 'geometry:shape'),
                'visual_geometry': ('color', 'road_context', 'roof_presence', 'semantic', 'geometry:size', 'geometry:shape')}
    all_components = ('color', 'size', 'shape', 'road_context', 'roof_presence', 'semantic', 'geometry:size', 'geometry:shape')
    variants['all'] = all_components
    variants['all_minus_color'] = tuple(x for x in all_components if x != 'color')
    variants['all_minus_size'] = tuple(x for x in all_components if x not in ('size', 'geometry:size'))
    variants['all_minus_shape'] = tuple(x for x in all_components if x not in ('shape', 'geometry:shape'))
    variants['all_minus_context'] = tuple(x for x in all_components if x not in ('road_context', 'roof_presence'))
    variants['all_minus_geometry'] = tuple(x for x in all_components if not x.startswith('geometry:'))

    def scored_rows(records, attributes, weight):
        output = []
        for record in records:
            goal = np.asarray(record['goal']) * (30 / 240)
            scored = []
            for candidate in record['candidates']:
                score = math.log(max(candidate['b0'], 1e-12))
                for attribute in attributes:
                    if attribute.startswith('geometry:'):
                        name = attribute.split(':', 1)[1]; required = record['required'].get(name)
                        if required: score += weight * (1.0 if candidate['geometry'][name] == required else -1.0)
                    else:
                        required = record['required'].get(attribute)
                        if required in candidate['visual'].get(attribute, {}): score += weight * math.log(max(candidate['visual'][attribute][required], 1e-6))
                distance = math.hypot(candidate['row'] - goal[0], candidate['col'] - goal[1]) * 410 / 30
                scored.append((score, distance))
            distances = np.asarray([distance for _, distance in sorted(scored, reverse=True)])
            row = {'top1_distance_m': float(distances[0]), 'oracle_distance@8_m': float(distances[:8].min()), 'oracle_distance@16_m': float(distances[:16].min())}
            for k in KS:
                for radius in RADII: row[f'recall@{k}/{int(radius)}m'] = float(distances[:k].min() <= radius)
            output.append(row)
        return output

    report = {'config': vars(args) | {'data_root': str(args.data_root), 'checkpoint': str(args.checkpoint)}, 'variants': {}}
    for name, attributes in variants.items():
        tuned = max(weights, key=lambda w: metric(scored_rows(all_results['val_seen']['records'], attributes, w))['recall@4/20m'])
        report['variants'][name] = {'weight': tuned}
        for split in ('val_seen', 'val_unseen'): report['variants'][name][split] = metric(scored_rows(all_results[split]['records'], attributes, tuned))
    report['timing'] = {split: value['inference_seconds'] for split, value in all_results.items()}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, default=str) + '\n')


if __name__ == '__main__': main()
