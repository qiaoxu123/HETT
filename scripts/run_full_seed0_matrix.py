"""Run the predeclared full seed-0 comparison matrix after the corrected baseline."""

import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
PREDECESSOR_UNIT = 'hett-corrected-baseline-full-s0-20260911'
PREDECESSOR_STATUS = ROOT / '00-control/runs/full_corrected_baseline_queue_s0_20260911/status.json'
CORRECTED = ROOT / '01-teacher-fix/runs/teacher_fix_full_s0'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def supervisor(worktree, run_name, *variant_args, phase='train-eval', checkpoint=None):
    directory = ROOT / worktree
    command = [PYTHON, '-u', str(directory / 'scripts/supervise_experiment.py'),
               '--python', PYTHON, '--run-dir', str(directory / 'runs' / run_name),
               '--epochs', '20', '--max-episodes', '0', '--save-every', '20',
               '--seed', '0', '--interval', '60']
    if phase == 'eval':
        command.extend(['--phase', 'eval', '--checkpoint', str(checkpoint)])
    command.extend(f'--variant-arg={value}' for value in variant_args)
    return command


def build_jobs():
    corrected_checkpoint = CORRECTED / 'checkpoints/best_val_unseen'
    grounding = ROOT / '03-grounding/runs/grounding_full_s0/checkpoints/best_val_unseen'
    return [
        ('recovery_off', supervisor(
            '02-recovery', 'recovery_off_full_eval_s0', '--disable_task_interaction',
            phase='eval', checkpoint=corrected_checkpoint)),
        ('recovery_on', supervisor(
            '02-recovery', 'recovery_on_full_eval_s0', '--disable_task_interaction',
            '--enable_stage_recovery', phase='eval', checkpoint=corrected_checkpoint)),
        ('grounding_disabled_control', supervisor(
            '03-grounding', 'grounding_disabled_full_eval_s0', '--disable_task_interaction',
            phase='eval', checkpoint=corrected_checkpoint)),
        ('grounding', supervisor(
            '03-grounding', 'grounding_full_s0', '--disable_task_interaction',
            '--enable_region_grounding')),
        ('combined', supervisor(
            '04-combined', 'combined_full_eval_s0', '--disable_task_interaction',
            '--enable_region_grounding', '--enable_stage_recovery',
            phase='eval', checkpoint=grounding)),
        ('hypothesis_disabled_control', supervisor(
            '05-hypotheses', 'hypothesis_disabled_full_eval_s0', '--disable_task_interaction',
            phase='eval', checkpoint=corrected_checkpoint)),
        ('hypothesis', supervisor(
            '05-hypotheses', 'hypothesis_full_s0', '--disable_task_interaction',
            '--enable_multi_hypothesis')),
        ('bidirectional', supervisor('06-bidir', 'bidir_full_s0')),
        ('paper_loss_only', supervisor(
            '07-loss-ablation', 'loss_paper_only_full_s0', '--disable_task_interaction',
            '--target_loss_weight', '0')),
        ('no_progress', supervisor(
            '07-loss-ablation', 'loss_no_progress_full_s0', '--disable_task_interaction',
            '--progress_loss_weight', '0')),
        ('neither_auxiliary', supervisor(
            '07-loss-ablation', 'loss_neither_full_s0', '--disable_task_interaction',
            '--progress_loss_weight', '0', '--target_loss_weight', '0')),
    ]


def analysis_command(output_dir):
    runs = {
        'corrected': CORRECTED,
        'recovery_off': ROOT / '02-recovery/runs/recovery_off_full_eval_s0',
        'recovery_on': ROOT / '02-recovery/runs/recovery_on_full_eval_s0',
        'grounding_control': ROOT / '03-grounding/runs/grounding_disabled_full_eval_s0',
        'grounding': ROOT / '03-grounding/runs/grounding_full_s0',
        'combined': ROOT / '04-combined/runs/combined_full_eval_s0',
        'hypothesis_control': ROOT / '05-hypotheses/runs/hypothesis_disabled_full_eval_s0',
        'hypothesis': ROOT / '05-hypotheses/runs/hypothesis_full_s0',
        'bidirectional': ROOT / '06-bidir/runs/bidir_full_s0',
        'paper_loss_only': ROOT / '07-loss-ablation/runs/loss_paper_only_full_s0',
        'no_progress': ROOT / '07-loss-ablation/runs/loss_no_progress_full_s0',
        'neither_auxiliary': ROOT / '07-loss-ablation/runs/loss_neither_full_s0',
    }
    command = [PYTHON, str(ROOT / '00-control/scripts/analyze_validation_matrix.py')]
    for name, path in runs.items():
        command.extend(['--run', f'{name}={path}'])
    for pair in [
            'corrected:recovery_off', 'recovery_off:recovery_on',
            'corrected:grounding_control', 'grounding_control:grounding',
            'grounding:combined', 'corrected:hypothesis_control',
            'hypothesis_control:hypothesis', 'corrected:bidirectional',
            'corrected:paper_loss_only', 'corrected:no_progress',
            'corrected:neither_auxiliary']:
        command.extend(['--pair', pair])
    command.extend(['--output-dir', str(output_dir)])
    return command


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
        raise SystemExit('corrected full baseline did not complete')

    jobs = build_jobs()
    write(args.run_dir / 'jobs.json', [{'name': name, 'command': command}
                                        for name, command in jobs])
    for name, command in jobs:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'running', 'job': name})
        append(args.run_dir / 'events.jsonl', {'time': stamp(), 'event': 'started', 'job': name})
        with (args.run_dir / f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT / '00-control', stdout=log,
                                    stderr=subprocess.STDOUT)
        append(args.run_dir / 'events.jsonl', {'time': stamp(), 'event': 'exited',
                                               'job': name, 'exit_code': result.returncode})
        if result.returncode:
            write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'failed',
                                                 'job': name, 'exit_code': result.returncode})
            raise SystemExit(result.returncode)

    analysis = analysis_command(args.run_dir / 'analysis')
    write(args.run_dir / 'analysis_command.json', analysis)
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
