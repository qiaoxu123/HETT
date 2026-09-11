"""Run the predeclared GPU smoke matrix serially after the active baseline."""

import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
BASELINE_STATUS = Path('/home/tenant2/Workspace/hett-crotonyl/runs/hett_baseline_fixed_20260911/status.json')


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def supervisor(worktree, run_name, *arguments):
    directory = ROOT / worktree
    return [PYTHON, '-u', str(directory / 'scripts/supervise_experiment.py'),
            '--python', PYTHON, '--run-dir', str(directory / 'runs' / run_name),
            '--epochs', '1', '--max-episodes', '4', '--save-every', '1',
            '--interval', '30', *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-dir', type=Path, required=True)
    args = parser.parse_args()
    args.queue_dir.mkdir(parents=True, exist_ok=False)
    teacher_run = ROOT / '01-teacher-fix/runs/teacher_fix_smoke_s0'
    grounding_run = ROOT / '03-grounding/runs/grounding_smoke_s0'
    jobs = [
        ('teacher_fix', supervisor(
            '01-teacher-fix', 'teacher_fix_smoke_s0',
            '--wait-for-unit', 'hett-baseline-20260911',
            '--require-status', str(BASELINE_STATUS),
            '--variant-arg=--disable_task_interaction')),
        ('recovery_off', supervisor(
            '02-recovery', 'recovery_off_smoke_s0', '--phase', 'eval',
            '--checkpoint', str(teacher_run / 'checkpoints/best_val_unseen'),
            '--variant-arg=--disable_task_interaction')),
        ('recovery_on', supervisor(
            '02-recovery', 'recovery_on_smoke_s0', '--phase', 'eval',
            '--checkpoint', str(teacher_run / 'checkpoints/best_val_unseen'),
            '--variant-arg=--disable_task_interaction',
            '--variant-arg=--enable_stage_recovery')),
        ('grounding', supervisor(
            '03-grounding', 'grounding_smoke_s0',
            '--variant-arg=--disable_task_interaction',
            '--variant-arg=--enable_region_grounding')),
        ('combined', supervisor(
            '04-combined', 'combined_smoke_s0', '--phase', 'eval',
            '--checkpoint', str(grounding_run / 'checkpoints/best_val_unseen'),
            '--variant-arg=--disable_task_interaction',
            '--variant-arg=--enable_region_grounding',
            '--variant-arg=--enable_stage_recovery')),
        ('multi_hypothesis', supervisor(
            '05-hypotheses', 'multi_hypothesis_smoke_s0',
            '--variant-arg=--disable_task_interaction',
            '--variant-arg=--enable_multi_hypothesis')),
    ]
    (args.queue_dir / 'jobs.json').write_text(json.dumps(
        [{'name': name, 'command': command} for name, command in jobs], indent=2) + '\n')
    for name, command in jobs:
        append(args.queue_dir / 'events.jsonl', {'time': stamp(), 'event': 'started', 'job': name})
        with (args.queue_dir / f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT / '00-control', stdout=log,
                                    stderr=subprocess.STDOUT)
        append(args.queue_dir / 'events.jsonl', {'time': stamp(), 'event': 'exited',
                                                 'job': name, 'exit_code': result.returncode})
        if result.returncode != 0:
            (args.queue_dir / 'status.json').write_text(json.dumps(
                {'time': stamp(), 'phase': 'failed', 'job': name,
                 'exit_code': result.returncode}, indent=2) + '\n')
            raise SystemExit(result.returncode)
    (args.queue_dir / 'status.json').write_text(json.dumps(
        {'time': stamp(), 'phase': 'complete'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
