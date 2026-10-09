#!/usr/bin/env python3
"""Compare same-checkpoint HETT GPU smoke outputs; never infer speed without measurements."""
import argparse
import json
from pathlib import Path


def read(directory, filename):
    return json.loads((Path(directory) / filename).read_text())


def compare(before, after):
    old_manifest = read(before, 'manifest.json')
    new_manifest = read(after, 'manifest.json')
    old_sha = old_manifest.get('initial_sha256')
    new_sha = new_manifest.get('initial_sha256')
    if not old_sha or old_sha != new_sha:
        raise ValueError('Different/missing initial checkpoint SHA-256; comparison is not controlled')
    rows = []
    for batch in (2, 8):
        old = read(before, f'gpu_batch{batch}.json')
        new = read(after, f'gpu_batch{batch}.json')
        if old['batch_size'] != batch or new['batch_size'] != batch:
            raise ValueError(f'Unexpected batch size in {batch} artifact')
        if old.get('optimizer_steps') != new.get('optimizer_steps'):
            raise ValueError(f'Optimizer-step counts differ at batch={batch}')
        baseline = old['seconds_per_batch']
        optimized = new['seconds_per_batch']
        rows.append({
            'batch': batch,
            'old_seconds': baseline,
            'new_seconds': optimized,
            'speedup': baseline / optimized,
            'old_gib': old['peak_vram_allocated_bytes'] / (1024 ** 3),
            'new_gib': new['peak_vram_allocated_bytes'] / (1024 ** 3),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = compare(args.before, args.after)
    lines = [
        '# Layered fastpath GPU smoke comparison',
        '',
        'Matched initialization SHA-256; same optimizer step counts.',
        'Confirm hardware, software, and episode sampling match separately.',
        'Times with profile_rollout=true MUST NOT be used for throughput claims.',
        '',
        '| Batch | Before seconds/batch | After seconds/batch | Speedup | Before VRAM GiB | After VRAM GiB |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for x in rows:
        lines.append(
            f"| {x['batch']} | {x['old_seconds']:.3f} | {x['new_seconds']:.3f} | "
            f"{x['speedup']:.3f}x | {x['old_gib']:.2f} | {x['new_gib']:.2f} |")
    lines.extend([
        '',
        'Short smoke throughput cannot demonstrate full-epoch speedup or navigation equivalence.',
        'Validate on full val_seen/val_unseen before reporting research results.',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
