"""Start the mandatory full corrected baseline only after every GPU smoke passes."""

import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
PREDECESSOR_UNIT = 'hett-loss-ablation-20260911'
PREDECESSOR_STATUS = ROOT / '00-control/runs/loss_ablation_queue_20260911/status.json'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def build_command():
    worktree = ROOT / '01-teacher-fix'
    return [PYTHON, '-u', str(worktree / 'scripts/supervise_experiment.py'),
            '--python', PYTHON,
            '--run-dir', str(worktree / 'runs/teacher_fix_full_s0'),
            '--epochs', '20', '--max-episodes', '0', '--save-every', '20',
            '--seed', '0', '--interval', '60',
            '--variant-arg=--disable_task_interaction']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    while subprocess.run(['systemctl', '--user', 'is-active', '--quiet',
                          PREDECESSOR_UNIT]).returncode == 0:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'waiting',
                                             'unit': PREDECESSOR_UNIT})
        time.sleep(60)
    predecessor = json.loads(PREDECESSOR_STATUS.read_text())
    if predecessor.get('phase') != 'complete':
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'blocked',
                                             'predecessor': predecessor})
        raise SystemExit('GPU smoke chain did not complete')

    command = build_command()
    write(args.run_dir / 'command.json', command)
    write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'running',
                                         'experiment': 'teacher_fix_full_s0'})
    with (args.run_dir / 'experiment.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT / '00-control', stdout=log,
                                stderr=subprocess.STDOUT)
    phase = 'complete' if result.returncode == 0 else 'failed'
    write(args.run_dir / 'status.json', {'time': stamp(), 'phase': phase,
                                         'exit_code': result.returncode,
                                         'experiment': 'teacher_fix_full_s0'})
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
