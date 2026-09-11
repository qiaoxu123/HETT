"""Run isolated real-data loss-weight smoke checks after bidirectional validation."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
PREDECESSOR_UNIT = 'hett-bidir-validation-20260911'
PREDECESSOR_STATUS = ROOT / '00-control/runs/bidir_validation_queue_20260911/status.json'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def supervisor(run_name, *loss_args):
    worktree = ROOT / '07-loss-ablation'
    command = [PYTHON, '-u', str(worktree / 'scripts/supervise_experiment.py'),
               '--python', PYTHON, '--run-dir', str(worktree / 'runs' / run_name),
               '--epochs', '1', '--max-episodes', '4', '--save-every', '1', '--interval', '30',
               '--variant-arg=--disable_task_interaction']
    command.extend(f'--variant-arg={value}' for value in loss_args)
    return command


def build_jobs():
    return [
        ('released', supervisor('loss_released_smoke_s0')),
        ('paper_loss_only', supervisor(
            'loss_paper_only_smoke_s0', '--target_loss_weight', '0')),
        ('no_progress', supervisor(
            'loss_no_progress_smoke_s0', '--progress_loss_weight', '0')),
        ('neither', supervisor(
            'loss_neither_smoke_s0', '--progress_loss_weight', '0',
            '--target_loss_weight', '0')),
    ]


def verify_run(run):
    status = json.loads((run / 'status.json').read_text())
    if status.get('phase') != 'complete':
        raise RuntimeError(f'incomplete run: {run}: {status}')
    checkpoint = run / 'checkpoints/best_val_unseen'
    metrics_path = run / 'checkpoints/epoch_metrics.jsonl'
    if not checkpoint.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f'missing checkpoint or metrics in {run}')
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f'expected one smoke epoch in {run}, got {len(rows)}')
    numeric = [value for key, value in rows[0].items()
               if key.endswith('_loss') and isinstance(value, (int, float))]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise RuntimeError(f'non-finite or missing losses in {run}: {numeric}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    while subprocess.run(['systemctl', '--user', 'is-active', '--quiet',
                          PREDECESSOR_UNIT]).returncode == 0:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'waiting',
                                             'unit': PREDECESSOR_UNIT})
        time.sleep(30)
    predecessor = json.loads(PREDECESSOR_STATUS.read_text())
    if predecessor.get('phase') != 'complete':
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'blocked',
                                             'predecessor': predecessor})
        raise SystemExit('bidirectional validation did not complete')

    jobs = build_jobs()
    write(args.run_dir / 'jobs.json', [{'name': name, 'command': command}
                                        for name, command in jobs])
    events = args.run_dir / 'events.jsonl'
    completed = {}
    for name, command in jobs:
        with events.open('a') as stream:
            stream.write(json.dumps({'time': stamp(), 'event': 'started', 'job': name}) + '\n')
        with (args.run_dir / f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT / '00-control', stdout=log,
                                    stderr=subprocess.STDOUT)
        with events.open('a') as stream:
            stream.write(json.dumps({'time': stamp(), 'event': 'exited', 'job': name,
                                     'exit_code': result.returncode}) + '\n')
        if result.returncode:
            write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'failed',
                                                  'job': name, 'exit_code': result.returncode})
            raise SystemExit(result.returncode)
        run = Path(command[command.index('--run-dir') + 1])
        verify_run(run)
        completed[name] = str(run)
    write(args.run_dir / 'summary.json', {
        'time': stamp(), 'purpose': 'runtime validation only; not performance evidence',
        'completed': completed,
    })
    write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'complete'})


if __name__ == '__main__':
    main()
