"""Run the leakage-controlled coarse-to-fine screening job on the shared GPU."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
WORKTREE = ROOT / '08-coarse-to-fine-target'
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
PREDECESSOR_UNIT = 'hett-corrected-baseline-full-s0-20260911.service'
PREDECESSOR_STATUS = ROOT / '00-control/runs/full_corrected_baseline_queue_s0_20260911/status.json'
DEFAULT_CORRECTED_CHECKPOINT = ROOT / '01-teacher-fix/runs/teacher_fix_full_s0/checkpoints/best_val_unseen'
GLOBAL_LOCK = ROOT / '.gpu-validation.lock'
INTERNAL_LOCK = ROOT / '.coarse-to-fine-internal.lock'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def active(unit):
    return subprocess.run(
        ['systemctl', '--user', 'is-active', '--quiet', unit],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def run_logged(command, log_path, cwd):
    with log_path.open('w') as log:
        return subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--parent-checkpoint', type=Path, default=DEFAULT_CORRECTED_CHECKPOINT)
    parser.add_argument('--parent-epoch', type=int, required=True,
                        help='completed epoch stored in parent checkpoint; screening adds exactly one epoch')
    parser.add_argument('--allow-paused-predecessor', action='store_true',
                        help='screen from a saved partial-baseline checkpoint after an intentional pause')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    parent_checkpoint = args.parent_checkpoint.resolve()
    if args.parent_epoch < 1:
        parser.error('--parent-epoch must be positive')
    run.mkdir(parents=True, exist_ok=False)
    write(run / 'protocol.json', {
        'time': stamp(),
        'seed': 0,
        'train_episodes': 512,
        'fine_tune_epochs': 1,
        'parent_epoch': args.parent_epoch,
        'total_epochs_argument': args.parent_epoch + 1,
        'parent_checkpoint': str(parent_checkpoint),
        'allow_paused_predecessor': args.allow_paused_predecessor,
        'test_unseen': False,
        'gates': {
            'first_cell_change_percent_min': 25.0,
            'mean_first_prediction_correct_advantage_m_min_exclusive': 0.0,
            'correct_first_prediction_better_percent_min': 55.0,
        },
    })

    GLOBAL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_LOCK.open('w') as lock_stream:
        write(run / 'status.json', {'time': stamp(), 'phase': 'waiting_for_gpu_lock'})
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        lock_stream.write(f'{Path(__file__).name} {run}\n')
        lock_stream.flush()

        while active(PREDECESSOR_UNIT) and not args.allow_paused_predecessor:
            write(run / 'status.json', {'time': stamp(), 'phase': 'waiting_for_predecessor',
                                         'unit': PREDECESSOR_UNIT})
            time.sleep(30)
        predecessor = json.loads(PREDECESSOR_STATUS.read_text())
        if predecessor.get('phase') != 'complete' and not args.allow_paused_predecessor:
            write(run / 'status.json', {'time': stamp(), 'phase': 'blocked',
                                         'predecessor': predecessor})
            raise SystemExit('corrected baseline did not complete successfully')
        if not parent_checkpoint.exists():
            raise FileNotFoundError(parent_checkpoint)

        train_run = run / 'finetune'
        command = [
            PYTHON, '-u', str(WORKTREE / 'scripts/supervise_experiment.py'),
            '--python', PYTHON,
            '--run-dir', str(train_run),
            '--epochs', str(args.parent_epoch + 1), '--max-episodes', '512', '--save-every', '1',
            '--seed', '0', '--interval', '30', '--lock-file', str(INTERNAL_LOCK),
            '--variant-arg=--disable_task_interaction',
            '--variant-arg=--coarse_to_fine_target',
            '--variant-arg=--checkpoint',
            f'--variant-arg={parent_checkpoint}',
        ]
        write(run / 'status.json', {'time': stamp(), 'phase': 'finetuning'})
        code = run_logged(command, run / 'finetune.log', WORKTREE)
        if code:
            write(run / 'status.json', {'time': stamp(), 'phase': 'failed',
                                         'step': 'finetune', 'exit_code': code})
            raise SystemExit(code)

        contrast_dir = run / 'hard_contrast'
        contrast_command = [
            PYTHON, str(train_run / 'source/scripts/evaluate_target_contrasts.py'),
            '--checkpoint', str(train_run / 'checkpoints/best_val_unseen'),
            '--output-dir', str(contrast_dir), '--pairs', '16', '--seed', '20260911',
        ]
        write(run / 'status.json', {'time': stamp(), 'phase': 'hard_contrast'})
        code = run_logged(contrast_command, run / 'hard_contrast.log', train_run / 'source')
        if code:
            write(run / 'status.json', {'time': stamp(), 'phase': 'failed',
                                         'step': 'hard_contrast', 'exit_code': code})
            raise SystemExit(code)

        metrics = json.loads((contrast_dir / 'summary.json').read_text())
        gates = {
            'cell_change': metrics['first_cell_change_percent'] >= 25.0,
            'positive_mean_advantage': metrics['mean_first_prediction_correct_advantage_m'] > 0.0,
            'majority_correct_better': metrics['correct_first_prediction_better_percent'] >= 55.0,
        }
        summary = {'time': stamp(), 'metrics': metrics, 'gates': gates,
                   'admit_full_seed0': all(gates.values())}
        write(run / 'summary.json', summary)
        write(run / 'status.json', {'time': stamp(), 'phase': 'complete',
                                     'admit_full_seed0': all(gates.values())})


if __name__ == '__main__':
    main()
