"""After prior smoke checks, run multi-hypothesis ablations and refresh reports."""

import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
PREDECESSOR_UNIT = 'hett-post-smoke-ablations-20260911'
PREDECESSOR_STATUS = ROOT / '00-control/runs/post_smoke_ablations_20260911/status.json'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def supervisor(run_name, *variant_args):
    worktree = ROOT / '05-hypotheses'
    checkpoint = worktree / 'runs/multi_hypothesis_smoke_s0/checkpoints/best_val_unseen'
    command = [PYTHON, '-u', str(worktree / 'scripts/supervise_experiment.py'),
               '--python', PYTHON, '--run-dir', str(worktree / 'runs' / run_name),
               '--phase', 'eval', '--checkpoint', str(checkpoint), '--epochs', '1',
               '--max-episodes', '4', '--save-every', '1', '--interval', '30']
    command.extend(f'--variant-arg={value}' for value in variant_args)
    return command


def build_jobs():
    return [
        ('hypothesis_off', supervisor('hypothesis_off_smoke_s0', '--disable_task_interaction')),
        ('hypothesis_no_temporal', supervisor(
            'hypothesis_no_temporal_smoke_s0', '--disable_task_interaction',
            '--enable_multi_hypothesis', '--hypothesis_decay', '0')),
    ]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    while subprocess.run(['systemctl', '--user', 'is-active', '--quiet', PREDECESSOR_UNIT]).returncode == 0:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'waiting',
                                             'unit': PREDECESSOR_UNIT})
        time.sleep(30)
    predecessor = json.loads(PREDECESSOR_STATUS.read_text())
    if predecessor.get('phase') != 'complete':
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'blocked',
                                             'predecessor': predecessor})
        raise SystemExit('predecessor ablations did not complete')

    jobs = build_jobs()
    write(args.run_dir / 'jobs.json', [{'name': name, 'command': command}
                                        for name, command in jobs])
    events = args.run_dir / 'events.jsonl'
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

    runs = {
        'hypothesis': ROOT / '05-hypotheses/runs/multi_hypothesis_smoke_s0',
        'hypothesis_off': ROOT / '05-hypotheses/runs/hypothesis_off_smoke_s0',
        'hypothesis_no_temporal': ROOT / '05-hypotheses/runs/hypothesis_no_temporal_smoke_s0',
    }
    analysis = [PYTHON, str(ROOT / '00-control/scripts/analyze_validation_matrix.py')]
    for name, path in runs.items():
        analysis += ['--run', f'{name}={path}']
    analysis += ['--pair', 'hypothesis:hypothesis_off',
                 '--pair', 'hypothesis:hypothesis_no_temporal',
                 '--output-dir', str(args.run_dir / 'analysis')]
    with (args.run_dir / 'analysis.log').open('w') as log:
        result = subprocess.run(analysis, cwd=ROOT / '00-control', stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'failed',
                                              'job': 'analysis', 'exit_code': result.returncode})
        raise SystemExit(result.returncode)
    write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'complete'})


if __name__ == '__main__':
    main()
