"""Run full seeds 17 and 42 for every claim after the seed-0 matrix succeeds."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
PREDECESSOR_UNIT = 'hett-full-seed0-matrix-20260911'
PREDECESSOR_STATUS = ROOT / '00-control/runs/full_seed0_matrix_20260911/status.json'
SEEDS = (17, 42)
MIN_FREE_GIB = 16


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def supervisor(worktree, run_name, seed, *variant_args, phase='train-eval', checkpoint=None):
    directory = ROOT / worktree
    command = [PYTHON, '-u', str(directory / 'scripts/supervise_experiment.py'),
               '--python', PYTHON, '--run-dir', str(directory / 'runs' / run_name),
               '--epochs', '20', '--max-episodes', '0', '--save-every', '20',
               '--seed', str(seed), '--interval', '60']
    if phase == 'eval':
        command.extend(['--phase', 'eval', '--checkpoint', str(checkpoint)])
    command.extend(f'--variant-arg={value}' for value in variant_args)
    return command


def jobs_for_seed(seed):
    baseline_run = ROOT / f'01-teacher-fix/runs/teacher_fix_full_s{seed}'
    baseline_checkpoint = baseline_run / 'checkpoints/best_val_unseen'
    grounding_checkpoint = ROOT / f'03-grounding/runs/grounding_full_s{seed}/checkpoints/best_val_unseen'
    return [
        (f'teacher_fix_s{seed}', supervisor(
            '01-teacher-fix', f'teacher_fix_full_s{seed}', seed,
            '--disable_task_interaction')),
        (f'recovery_off_s{seed}', supervisor(
            '02-recovery', f'recovery_off_full_eval_s{seed}', seed,
            '--disable_task_interaction', phase='eval', checkpoint=baseline_checkpoint)),
        (f'recovery_on_s{seed}', supervisor(
            '02-recovery', f'recovery_on_full_eval_s{seed}', seed,
            '--disable_task_interaction', '--enable_stage_recovery',
            phase='eval', checkpoint=baseline_checkpoint)),
        (f'grounding_s{seed}', supervisor(
            '03-grounding', f'grounding_full_s{seed}', seed,
            '--disable_task_interaction', '--enable_region_grounding')),
        (f'combined_s{seed}', supervisor(
            '04-combined', f'combined_full_eval_s{seed}', seed,
            '--disable_task_interaction', '--enable_region_grounding',
            '--enable_stage_recovery', phase='eval', checkpoint=grounding_checkpoint)),
        (f'hypothesis_s{seed}', supervisor(
            '05-hypotheses', f'hypothesis_full_s{seed}', seed,
            '--disable_task_interaction', '--enable_multi_hypothesis')),
        (f'bidirectional_s{seed}', supervisor(
            '06-bidir', f'bidir_full_s{seed}', seed)),
        (f'paper_loss_only_s{seed}', supervisor(
            '07-loss-ablation', f'loss_paper_only_full_s{seed}', seed,
            '--disable_task_interaction', '--target_loss_weight', '0')),
        (f'no_progress_s{seed}', supervisor(
            '07-loss-ablation', f'loss_no_progress_full_s{seed}', seed,
            '--disable_task_interaction', '--progress_loss_weight', '0')),
        (f'neither_auxiliary_s{seed}', supervisor(
            '07-loss-ablation', f'loss_neither_full_s{seed}', seed,
            '--disable_task_interaction', '--progress_loss_weight', '0',
            '--target_loss_weight', '0')),
    ]


def build_jobs():
    return [job for seed in SEEDS for job in jobs_for_seed(seed)]


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
        raise SystemExit('full seed-0 matrix did not complete')

    jobs = build_jobs()
    write(args.run_dir / 'jobs.json', [{'name': name, 'command': command}
                                        for name, command in jobs])
    for name, command in jobs:
        free_gib = shutil.disk_usage(ROOT).free / 2**30
        if free_gib < MIN_FREE_GIB:
            write(args.run_dir / 'status.json', {
                'time': stamp(), 'phase': 'blocked', 'job': name,
                'reason': f'only {free_gib:.2f} GiB free; requires {MIN_FREE_GIB} GiB'})
            raise SystemExit('insufficient disk for next full training')
        write(args.run_dir / 'status.json', {
            'time': stamp(), 'phase': 'running', 'job': name, 'disk_free_gib': free_gib})
        append(args.run_dir / 'events.jsonl', {
            'time': stamp(), 'event': 'started', 'job': name, 'disk_free_gib': free_gib})
        with (args.run_dir / f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT / '00-control', stdout=log,
                                    stderr=subprocess.STDOUT)
        append(args.run_dir / 'events.jsonl', {'time': stamp(), 'event': 'exited',
                                               'job': name, 'exit_code': result.returncode})
        if result.returncode:
            write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'failed',
                                                 'job': name, 'exit_code': result.returncode})
            raise SystemExit(result.returncode)
    write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'complete',
                                         'seeds': list(SEEDS)})


if __name__ == '__main__':
    main()
