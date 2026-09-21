"""Paired validation of bbox centres and a 10 m target grid on val_unseen."""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/home/tenant2/Workspace/hett-crotonyl')
GEOM = Path('/home/tenant2/Workspace/hett-experiments/22-language-geometry')
sys.path.insert(0, str(REPO))
from multiagent.observation import cropclient  # noqa: E402
from multiagent.space import Pose4D  # noqa: E402

MODEL = '/home/tenant2/dataext/models/Qwen2.5-VL-3B-Instruct'
ADAPTER = GEOM / 'runs/grounding/lora/step3000'
INDEX = GEOM / 'datasets/grounding_v1/val_unseen.jsonl'
SIZE = 896
HALF = 50.0
GRID = 10
SEED = 2501

BBOX_PROMPT = (
    '{description}\nThis is a top-down aerial view covering 100 by 100 metres. '
    'Locate the object the sentence refers to. Answer with its bounding box in image '
    'pixels, as JSON: {{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}'
)
GRID_PROMPT = (
    '{description}\nThis is a top-down aerial view covering 100 by 100 metres, divided '
    'into a 10 by 10 grid. Rows are numbered 0 to 9 from top to bottom; columns are '
    'numbered 0 to 9 from left to right. Locate the referred object and answer only as '
    'JSON: {{"grid_rc": [row, column]}}'
)


def parse_box(text):
    match = re.search(r'"?bbox_2d"?\s*:\s*\[([^]]+)\]', text)
    if not match:
        return None
    values = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', match.group(1))]
    return values if len(values) == 4 else None


def parse_grid(text):
    match = re.search(r'"?grid_rc"?\s*:\s*\[([^]]+)\]', text)
    if not match:
        return None
    values = [int(v) for v in re.findall(r'-?\d+', match.group(1))]
    if len(values) != 2 or not all(0 <= v < GRID for v in values):
        return None
    return values


def pixel_to_world(pose, col, row):
    gsd = 2 * HALF / SIZE
    left, front = (col - SIZE / 2) * gsd, (SIZE / 2 - row) * gsd
    cos, sin = np.cos(pose.yaw), np.sin(pose.yaw)
    return pose.x + front * cos - left * sin, pose.y + front * sin + left * cos


def error_m(pose, col, row, target):
    point = pixel_to_world(pose, col, row)
    return float(np.hypot(point[0] - target[0], point[1] - target[1]))


def quantize(col, row):
    cell = SIZE / GRID
    c = min(GRID - 1, max(0, int(col / cell)))
    r = min(GRID - 1, max(0, int(row / cell)))
    return r, c, (c + .5) * cell, (r + .5) * cell


def relation_type(text):
    s = text.lower()
    if any(w in s for w in ['between', 'middle of', 'in between']):
        return 'between'
    if re.search(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|\d+(?:st|nd|rd|th))\b', s):
        return 'ordinal'
    if any(w in s for w in ['left', 'right', 'front', 'behind', 'north', 'south', 'east', 'west']):
        return 'directional'
    return 'other'


def select_records(per_map):
    records = [json.loads(line) for line in INDEX.open()]
    eligible = [r for r in records if r['in_view'] and r['target_on_data']]
    episodes = defaultdict(list)
    for record in eligible:
        episodes[tuple(record['episode'])].append(record)
    midpoint = [min(rows, key=lambda r: abs(r['trajectory_fraction'] - .5))
                for rows in episodes.values()]
    maps = defaultdict(list)
    for record in midpoint:
        maps[record['map_name']].append(record)
    chosen = []
    for map_name in sorted(maps):
        rows = sorted(maps[map_name], key=lambda r: tuple(r['episode']))
        random.Random(f'{SEED}:{map_name}').shuffle(rows)
        if len(rows) < per_map:
            raise RuntimeError(f'{map_name}: only {len(rows)} unique eligible episodes')
        chosen.extend(rows[:per_map])
    return chosen


def generate(model, processor, process_vision_info, image, prompt):
    from PIL import Image
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': Image.fromarray(image)},
        {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, return_tensors='pt').to('cuda')
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    return processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)[0]


def metrics(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        'n': len(array), 'valid': len(finite),
        'invalid_rate': round(100 * (1 - len(finite) / len(array)), 2),
        'hit10': round(100 * float(np.mean(array <= 10)), 2),
        'hit20': round(100 * float(np.mean(array <= 20)), 2),
        'median_m': round(float(np.median(finite)), 2) if len(finite) else None,
        'p90_m': round(float(np.percentile(finite, 90)), 2) if len(finite) else None,
        'mean_m': round(float(np.mean(finite)), 2) if len(finite) else None,
    }


def write_report(path, result):
    labels = [('bbox_center', 'bbox 中心'), ('bbox_quantized_10m', 'bbox 中心→10 m 网格'),
              ('direct_grid_10m', '直接输出 10 m 网格')]
    lines = ['# 10 m 粗网格目标定位验证', '',
             f"样本：val_unseen，{result['sample_count']} 个不同 episode，"
             f"{result['maps']} 张地图。主指标为 Hit@20。", '',
             '| 方法 | Hit@20 | Hit@10 | 中位误差 | P90 | 无效输出 |',
             '|---|---:|---:|---:|---:|---:|']
    for key, label in labels:
        m = result['metrics'][key]
        lines.append(f"| {label} | {m['hit20']:.2f}% | {m['hit10']:.2f}% | "
                     f"{m['median_m']} m | {m['p90_m']} m | {m['invalid_rate']:.2f}% |")
    delta = result['metrics']['bbox_quantized_10m']['hit20'] - result['metrics']['bbox_center']['hit20']
    lines += ['', f'量化相对原始中心的 Hit@20 变化：**{delta:+.2f} 个百分点**。', '',
              '## 分关系类型 Hit@20', '',
              '| 类型 | 数量 | bbox 中心 | 量化网格 | 直接网格 |', '|---|---:|---:|---:|---:|']
    for kind, row in result['by_relation'].items():
        lines.append(f"| {kind} | {row['n']} | {row['bbox_center']:.1f}% | "
                     f"{row['bbox_quantized_10m']:.1f}% | {row['direct_grid_10m']:.1f}% |")
    passed = delta >= -3
    lines += ['', '## 判定', '',
              ('**通过**：10 m 网格表示没有造成超过 3 个百分点的 Hit@20 损失。'
               if passed else
               '**未通过**：10 m 网格表示造成了超过 3 个百分点的 Hit@20 损失。'),
              '', '直接网格是输出接口零样本迁移诊断；它没有接受热力图或网格监督。']
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--per-map', type=int, default=20)
    args = parser.parse_args()
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    records = select_records(args.per_map)
    cropclient.load_image_cache()
    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER).cuda().eval()
    samples = []
    started = time.time()
    for index, record in enumerate(records, 1):
        pose = Pose4D(*record['pose'])
        image = cropclient.crop_image(record['map_name'], pose, (SIZE, SIZE), 'rgb')
        bbox_reply = generate(model, processor, process_vision_info, image,
                              BBOX_PROMPT.format(description=record['description']))
        grid_reply = generate(model, processor, process_vision_info, image,
                              GRID_PROMPT.format(description=record['description']))
        box, grid = parse_box(bbox_reply), parse_grid(grid_reply)
        raw_error = quant_error = grid_error = float('inf')
        raw_center = quant_cell = None
        if box and box[2] > box[0] and box[3] > box[1]:
            col, row = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            raw_center = [col, row]
            raw_error = error_m(pose, col, row, record['target_xy'])
            qr, qc, qcol, qrow = quantize(col, row)
            quant_cell = [qr, qc]
            quant_error = error_m(pose, qcol, qrow, record['target_xy'])
        if grid:
            row, col = grid
            cell = SIZE / GRID
            grid_error = error_m(pose, (col + .5) * cell, (row + .5) * cell,
                                 record['target_xy'])
        samples.append({
            'episode': record['episode'], 'map_name': record['map_name'],
            'description': record['description'], 'relation_type': relation_type(record['description']),
            'bbox_reply': bbox_reply, 'bbox': box, 'bbox_center': raw_center,
            'bbox_quantized_cell': quant_cell, 'grid_reply': grid_reply, 'direct_grid_cell': grid,
            'error_m': {'bbox_center': raw_error if np.isfinite(raw_error) else None,
                        'bbox_quantized_10m': quant_error if np.isfinite(quant_error) else None,
                        'direct_grid_10m': grid_error if np.isfinite(grid_error) else None},
        })
        if index % 10 == 0:
            print(f'{index}/{len(records)} elapsed={time.time()-started:.0f}s', flush=True)

    keys = ['bbox_center', 'bbox_quantized_10m', 'direct_grid_10m']
    errors = {key: [s['error_m'][key] if s['error_m'][key] is not None else float('inf')
                    for s in samples] for key in keys}
    by_relation = {}
    for kind in sorted({s['relation_type'] for s in samples}):
        subset = [s for s in samples if s['relation_type'] == kind]
        row = {'n': len(subset)}
        for key in keys:
            row[key] = round(100 * np.mean([(s['error_m'][key] if s['error_m'][key] is not None
                                             else float('inf')) <= 20
                                             for s in subset]), 2)
        by_relation[kind] = row
    result = {
        'protocol': 'PROTOCOL.md', 'model': MODEL, 'adapter': str(ADAPTER),
        'split': 'val_unseen', 'sample_count': len(samples),
        'maps': len({s['map_name'] for s in samples}), 'seed': SEED,
        'elapsed_seconds': round(time.time() - started, 1),
        'metrics': {key: metrics(errors[key]) for key in keys},
        'by_relation': by_relation, 'samples': samples,
    }
    (args.run_dir / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                          allow_nan=False))
    write_report(args.run_dir / 'REPORT.md', result)
    print(json.dumps(result['metrics'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
