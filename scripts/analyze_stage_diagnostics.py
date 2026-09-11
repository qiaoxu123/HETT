"""CPU-only diagnostics; aggregate logs cannot establish training causality."""
import argparse
import json
import re
from pathlib import Path


def historical(path):
    rows = []
    epoch = None
    stages = []
    split_index = 0
    for line in path.read_text().splitlines():
        if line.startswith('stage '):
            stages.append([float(x) for x in line.split()[1:]])
        if line.startswith('epoch '):
            epoch = int(line.split()[1])
            split_index = 0
        match = re.match(r'(val_seen|val_unseen|test_unseen)\s*,(.*)', line)
        if not match:
            continue
        values = {key: float(value) for key, value in
                  re.findall(r'(\w+):\s*([-+\d.]+)', match[2])}
        fine_hit = values['oracle_sr2']
        lost = fine_hit - values['sr']
        row = dict(epoch=epoch, split=match[1], **values)
        row.update(fine_hit_then_lost_pp=round(lost, 4),
                   lost_percent_of_fine_hits=round(100 * lost / fine_hit, 2),
                   fine_net_sr_gain_pp=round(values['sr'] - values['sr1'], 4),
                   fine_ne_change_m=round(values['ne'] - values['stage1_ne'], 4))
        if len(stages) >= 3 and split_index < 3:
            coarse, forward, rotate = stages[-3:][split_index]
            row.update(coarse_rounds=coarse, fine_forward_rounds=forward,
                       fine_rotation_rounds=rotate,
                       fine_rounds=round(forward + rotate, 4))
        rows.append(row)
        split_index += 1
    return rows


def trajectories(path, threshold):
    # Load only trusted local predictions: these contain Python pose objects.
    import torch
    predictions = torch.load(path, map_location='cpu', weights_only=False)
    rows = []
    for key, item in predictions.items():
        full = item['trajectory']
        coarse = item['stage1_trajectory']
        fine = item['stage2_trajectory']
        goal = item['goal']
        distance = lambda pose: float(pose.xy.dist_to(goal))
        start = fine[0] if fine else full[-1]
        rows.append(dict(id=str(key), entered_fine=bool(fine),
                         coarse_rounds=max(0, len(coarse)-1),
                         fine_rounds=max(0, len(fine)-1),
                         fine_start_distance=distance(start),
                         final_distance=distance(full[-1]),
                         fine_hit=any(distance(p) <= threshold for p in fine),
                         success=distance(full[-1]) <= threshold))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--threshold', type=float, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.log) == bool(args.predictions):
        parser.error('choose exactly one of --log or --predictions')
    rows = historical(args.log) if args.log else trajectories(args.predictions, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(rows if args.log else {'episodes': len(rows)}, indent=2))
