"""Run real-data bidirectional-attention checks after the other GPU smoke jobs."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
PREDECESSOR_UNIT = 'hett-grounding-contrast-20260911'
PREDECESSOR_STATUS = ROOT / '00-control/runs/grounding_contrast_queue_20260911/status.json'
LOCK = ROOT / '.gpu-validation.lock'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


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
        raise SystemExit('predecessor contrast did not complete')

    worktree = ROOT / '06-bidir'
    gradient_output = worktree / 'runs/bidir_gradient_smoke_s0'
    gradient_output.mkdir(parents=True, exist_ok=False)
    check = [PYTHON, '-u', str(worktree / 'multiagent/check_single_gpu.py'),
             '--max_episodes', '4', '--batch_size', '2', '--grad_accum', '4',
             '--output_dir', str(gradient_output)]
    train_run = worktree / 'runs/bidir_smoke_s0'
    train = [PYTHON, '-u', str(worktree / 'scripts/supervise_experiment.py'),
             '--python', PYTHON, '--run-dir', str(train_run), '--epochs', '1',
             '--max-episodes', '4', '--save-every', '1', '--interval', '30']
    write(args.run_dir / 'commands.json', {'gradient_check': check, 'train_eval': train})
    lock_stream = LOCK.open('w')
    fcntl.flock(lock_stream, fcntl.LOCK_EX)
    with (args.run_dir / 'gradient_check.log').open('w') as log:
        result = subprocess.run(check, cwd=worktree, stdout=log, stderr=subprocess.STDOUT)
    fcntl.flock(lock_stream, fcntl.LOCK_UN)
    lock_stream.close()
    if result.returncode:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'failed',
                                              'job': 'gradient_check', 'exit_code': result.returncode})
        raise SystemExit(result.returncode)
    with (args.run_dir / 'train_eval.log').open('w') as log:
        result = subprocess.run(train, cwd=worktree, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'failed',
                                              'job': 'train_eval', 'exit_code': result.returncode})
        raise SystemExit(result.returncode)

    analysis = [PYTHON, str(ROOT / '00-control/scripts/analyze_validation_matrix.py'),
                '--run', f"no_interaction={ROOT / '01-teacher-fix/runs/teacher_fix_smoke_s0'}",
                '--run', f'bidirectional={train_run}', '--pair', 'no_interaction:bidirectional',
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
