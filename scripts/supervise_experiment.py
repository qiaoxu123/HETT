#!/usr/bin/env python3
"""Run an isolated HETT experiment with a source snapshot and GPU lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def _stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def _json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(run):
    target = run / 'source'
    target.mkdir()
    for directory in ('multiagent', 'scripts', 'tests', 'docs'):
        if not (ROOT / directory).exists():
            continue
        shutil.copytree(
            ROOT / directory,
            target / directory,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
        )
    for name in ('data', 'weights'):
        (target / name).symlink_to((ROOT / name).resolve(), target_is_directory=True)
    return target


def _command(python, mode, checkpoint, variants):
    command = [
        python,
        'main.py',
        '--mode', mode,
        '--world_size', '1',
        '--batch_size', '2',
        '--grid_size', '7',
        '--max_action_len', '20',
        '--seed', '0',
    ]
    if checkpoint:
        command += ['--checkpoint', str(Path(checkpoint).resolve())]
    return command + variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--python', required=True)
    parser.add_argument('--phase', choices=('train', 'eval', 'train-eval', 'static-belief'), default='train-eval')
    parser.add_argument('--dataset-root', type=Path, help='required for --phase grounding')
    parser.add_argument('--pointcloud-root', type=Path, help='required for --phase pointcloud-viewer')
    parser.add_argument('--checkpoint')
    parser.add_argument('--variant-arg', action='append', default=[])
    parser.add_argument(
        '--lock-file',
        type=Path,
        default=Path('/home/ubuntu5/Workspace/hett-experiments/.gpu-validation.lock'),
    )
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    source = _snapshot(run)
    git_head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    git_status = subprocess.run(
        ['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    (run / 'git_status.txt').write_text(git_status)
    (run / 'git.diff').write_text(
        subprocess.run(
            ['git', 'diff', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    )
    hashes = {
        str(path.relative_to(source)): _sha256(path)
        for path in source.rglob('*.py')
    }
    _json(
        run / 'provenance.json',
        {
            'started': _stamp(),
            'git_head': git_head,
            'git_status': git_status.splitlines(),
            'phase': args.phase,
            'checkpoint': args.checkpoint,
            'variant_args': args.variant_arg,
            'source_sha256': hashes,
        },
    )
    checkpoint_dir = run / 'checkpoints'
    checkpoint_dir.mkdir()
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES='0',
        PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        HETT_CHECKPOINT_DIR=str(checkpoint_dir),
    )
    phases = []
    if args.phase == 'static-belief':
        phases.append((
            'static_belief',
            [args.python, 'scripts/train_static_belief.py',
             '--data-root', str((source / 'data').resolve()),
             '--output', str((run / 'artifacts').resolve()), *args.variant_arg],
        ))
    else:
        if args.phase in ('train', 'train-eval'):
            phases.append(('training', _command(args.python, 'train', args.checkpoint, args.variant_arg)))
        if args.phase in ('eval', 'train-eval'):
            checkpoint = args.checkpoint or checkpoint_dir / 'best_val_unseen'
            phases.append(('evaluation', _command(args.python, 'eval', checkpoint, args.variant_arg)))
    _json(run / 'commands.json', {name: command for name, command in phases})

    cpu_only = False
    lock_path = run / 'cpu-only.lock' if cpu_only else args.lock_file
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        _json(run / 'status.json', {'time': _stamp(), 'phase': 'waiting_cpu' if cpu_only else 'waiting_gpu'})
        if not cpu_only:
            fcntl.flock(lock, fcntl.LOCK_EX)
        lock.write(f'{os.getpid()} {run}\n')
        lock.flush()
        for name, command in phases:
            _json(run / 'status.json', {'time': _stamp(), 'phase': name})
            with (run / f'{name}.log').open('w') as log:
                process = subprocess.Popen(
                    command,
                    cwd=source if args.phase == 'static-belief' else source / 'multiagent',
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while process.poll() is None:
                    telemetry = 'CPU-only' if cpu_only else subprocess.run(
                        [
                            'nvidia-smi',
                            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                            '--format=csv,noheader,nounits',
                        ],
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                    _json(
                        run / 'status.json',
                        {'time': _stamp(), 'phase': name, 'pid': process.pid, 'gpu': telemetry},
                    )
                    time.sleep(5)
                if process.returncode:
                    _json(
                        run / 'status.json',
                        {'time': _stamp(), 'phase': 'failed', 'failed_phase': name, 'exit_code': process.returncode},
                    )
                    raise SystemExit(process.returncode)
        _json(run / 'status.json', {'time': _stamp(), 'phase': 'complete'})


if __name__ == '__main__':
    main()
