"""Resume independent evaluation and hard contrast after a completed screening train."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
WORKTREE = ROOT / '08-coarse-to-fine-target'
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
GLOBAL_LOCK = ROOT / '.gpu-validation.lock'
INTERNAL_LOCK = ROOT / '.coarse-to-fine-posttrain-internal.lock'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def run_logged(command, log_path, cwd):
    with log_path.open('w') as log:
        return subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    run.mkdir(parents=True, exist_ok=False)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    with GLOBAL_LOCK.open('w') as lock_stream:
        write(run / 'status.json', {'time': stamp(), 'phase': 'waiting_for_gpu_lock'})
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        evaluation = run / 'evaluation'
        command = [
            PYTHON, '-u', str(WORKTREE / 'scripts/supervise_experiment.py'),
            '--python', PYTHON, '--run-dir', str(evaluation), '--phase', 'eval',
            '--checkpoint', str(checkpoint), '--max-episodes', '512', '--seed', '0',
            '--interval', '30', '--lock-file', str(INTERNAL_LOCK),
            '--variant-arg=--disable_task_interaction',
            '--variant-arg=--coarse_to_fine_target',
        ]
        write(run / 'status.json', {'time': stamp(), 'phase': 'evaluation'})
        code = run_logged(command, run / 'evaluation.log', WORKTREE)
        if code:
            write(run / 'status.json', {'time': stamp(), 'phase': 'failed',
                                         'step': 'evaluation', 'exit_code': code})
            raise SystemExit(code)

        contrast = run / 'hard_contrast'
        command = [
            PYTHON, str(evaluation / 'source/scripts/evaluate_target_contrasts.py'),
            '--checkpoint', str(checkpoint), '--output-dir', str(contrast),
            '--pairs', '16', '--seed', '20260911',
        ]
        write(run / 'status.json', {'time': stamp(), 'phase': 'hard_contrast'})
        code = run_logged(command, run / 'hard_contrast.log', evaluation / 'source')
        if code:
            write(run / 'status.json', {'time': stamp(), 'phase': 'failed',
                                         'step': 'hard_contrast', 'exit_code': code})
            raise SystemExit(code)

        metrics = json.loads((contrast / 'summary.json').read_text())
        gates = {
            'cell_change': metrics['first_cell_change_percent'] >= 25.0,
            'positive_mean_advantage': metrics['mean_first_prediction_correct_advantage_m'] > 0.0,
            'majority_correct_better': metrics['correct_first_prediction_better_percent'] >= 55.0,
        }
        write(run / 'summary.json', {'time': stamp(), 'metrics': metrics, 'gates': gates,
                                      'admit_full_seed0': all(gates.values())})
        write(run / 'status.json', {'time': stamp(), 'phase': 'complete',
                                     'admit_full_seed0': all(gates.values())})


if __name__ == '__main__':
    main()
