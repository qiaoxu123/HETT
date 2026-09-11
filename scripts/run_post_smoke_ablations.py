"""After the smoke queue, run grounding ablations and generate paired reports."""

import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
QUEUE_UNIT = 'hett-validation-queue-20260911'
QUEUE_STATUS = ROOT / '00-control/runs/gpu_validation_queue_20260911/status.json'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def supervisor(run_name, *variant_args):
    worktree = ROOT / '03-grounding'
    checkpoint = worktree / 'runs/grounding_smoke_s0/checkpoints/best_val_unseen'
    command = [PYTHON, '-u', str(worktree / 'scripts/supervise_experiment.py'),
               '--python', PYTHON, '--run-dir', str(worktree / 'runs' / run_name),
               '--phase', 'eval', '--checkpoint', str(checkpoint),
               '--epochs', '1', '--max-episodes', '4', '--save-every', '1', '--interval', '30']
    command.extend(f'--variant-arg={value}' for value in variant_args)
    return command


def build_jobs():
    return [
        ('grounding_off', supervisor('grounding_off_smoke_s0', '--disable_task_interaction')),
        ('grounding_shuffle_language', supervisor(
            'grounding_shuffle_language_smoke_s0', '--disable_task_interaction',
            '--enable_region_grounding', '--grounding_ablation', 'shuffle_language')),
        ('grounding_shuffle_visual', supervisor(
            'grounding_shuffle_visual_smoke_s0', '--disable_task_interaction',
            '--enable_region_grounding', '--grounding_ablation', 'shuffle_visual')),
    ]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    while subprocess.run(['systemctl', '--user', 'is-active', '--quiet', QUEUE_UNIT]).returncode == 0:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'waiting', 'unit': QUEUE_UNIT})
        time.sleep(30)
    predecessor = json.loads(QUEUE_STATUS.read_text())
    if predecessor.get('phase') != 'complete':
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'blocked',
                                             'predecessor': predecessor})
        raise SystemExit('smoke queue did not complete')

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
        'teacher': ROOT / '01-teacher-fix/runs/teacher_fix_smoke_s0',
        'recovery_off': ROOT / '02-recovery/runs/recovery_off_smoke_s0',
        'recovery_on': ROOT / '02-recovery/runs/recovery_on_smoke_s0',
        'grounding': ROOT / '03-grounding/runs/grounding_smoke_s0',
        'grounding_off': ROOT / '03-grounding/runs/grounding_off_smoke_s0',
        'shuffle_language': ROOT / '03-grounding/runs/grounding_shuffle_language_smoke_s0',
        'shuffle_visual': ROOT / '03-grounding/runs/grounding_shuffle_visual_smoke_s0',
        'combined': ROOT / '04-combined/runs/combined_smoke_s0',
        'hypothesis': ROOT / '05-hypotheses/runs/multi_hypothesis_smoke_s0',
    }
    analysis = [PYTHON, str(ROOT / '00-control/scripts/analyze_validation_matrix.py')]
    for name, path in runs.items():
        analysis += ['--run', f'{name}={path}']
    analysis += ['--pair', 'recovery_off:recovery_on',
                 '--pair', 'grounding:combined',
                 '--pair', 'grounding:grounding_off',
                 '--pair', 'grounding:shuffle_language',
                 '--pair', 'grounding:shuffle_visual',
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
