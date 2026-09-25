"""Run one isolated HETT train/eval experiment with provenance and telemetry."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def digest(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def command_output(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
    return result.stdout


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def append_jsonl(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-episodes', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--variant-arg', action='append', default=[])
    parser.add_argument(
        '--lock-file',
        default='/home/rental/20260922_1/Workspace/hett-experiments/.gpu-validation.lock',
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.save_every < 1 or args.max_episodes < 0:
        parser.error('epochs/save-every must be positive and max-episodes nonnegative')

    root = Path(__file__).resolve().parent.parent
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / 'source'
    snapshot.mkdir()
    for directory in ['multiagent', 'gsamllavanav', 'vlnce', 'scripts', 'tests']:
        source = root / directory
        if source.exists():
            shutil.copytree(
                source,
                snapshot / directory,
                ignore=shutil.ignore_patterns('checkpoints', '__pycache__', '*.pyc'),
            )
    for name in [
        'README.md', 'SINGLE_GPU.md', 'requirements.txt', 'CHANGELOG.md',
        'TARGET_BELIEF_PROTOCOL.md',
    ]:
        if (root / name).exists():
            shutil.copy2(root / name, snapshot / name)
    for name in ['data', 'weights']:
        (snapshot / name).symlink_to((root / name).resolve(), target_is_directory=True)

    (run / 'git.diff').write_text(command_output(['git', 'diff', 'HEAD'], root))
    (run / 'git_status.txt').write_text(command_output(['git', 'status', '--short'], root))
    (run / 'pip_freeze.txt').write_text(command_output([args.python, '-m', 'pip', 'freeze']))
    (run / 'gpu_initial.txt').write_text(command_output(['nvidia-smi']))

    input_paths = [
        root / 'data/cityrefer/objects.json',
        root / 'data/cityrefer/processed_descriptions.json',
        root / 'weights/best.pt',
        root / 'weights/yolo_v3.cfg',
    ]
    input_paths += sorted((root / 'data/processed_citynav').glob('citynav_*.json'))
    input_hashes = {
        str(path): {
            'resolved': str(path.resolve()),
            'bytes': path.stat().st_size,
            'sha256': digest(path),
        }
        for path in input_paths
    }
    source_hashes = {
        str(path.relative_to(snapshot)): digest(path)
        for path in snapshot.rglob('*.py')
    }

    checkpoint_dir = run / 'checkpoints'
    common = [
        '--seed', str(args.seed),
        '--max_episodes', str(args.max_episodes),
    ] + list(args.variant_arg)
    train = [
        'bash', str(snapshot / 'multiagent/train.sh'),
        '--epochs', str(args.epochs),
        '--save_every', str(args.save_every),
        '--output_dir', str(checkpoint_dir),
    ] + common
    evaluation = [
        'bash', str(snapshot / 'multiagent/eval.sh'),
        '--checkpoint', str(checkpoint_dir / 'best_val_unseen'),
        '--output_dir', str(run / 'evaluation'),
    ] + common
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES='0',
        PYTHONUNBUFFERED='1',
        PYTHON=args.python,
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
    )
    atomic_json(run / 'provenance.json', {
        'started': stamp(),
        'git_head': command_output(['git', 'rev-parse', 'HEAD'], root).strip(),
        'git_branch': command_output(['git', 'branch', '--show-current'], root).strip(),
        'source_sha256': source_hashes,
        'input_sha256': input_hashes,
        'seed': args.seed,
        'epochs': args.epochs,
        'variant_args': args.variant_arg,
        'note': 'multi-landmark HETT belief head; fixed-height official 20-step protocol',
    })
    atomic_json(run / 'commands.json', {
        'train': train,
        'evaluation': evaluation,
        'environment_overrides': {
            key: environment[key]
            for key in [
                'CUDA_VISIBLE_DEVICES', 'PYTHONUNBUFFERED', 'PYTHON',
                'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
            ]
        },
    })

    lock_path = Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open('w')
    atomic_json(run / 'status.json', {
        'time': stamp(), 'phase': 'waiting_gpu', 'lock_file': str(lock_path),
    })
    fcntl.flock(lock_stream, fcntl.LOCK_EX)
    lock_stream.write(f'{os.getpid()} {run}\n')
    lock_stream.flush()

    stopped = False
    child = None

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    archived = set()

    def archive_checkpoints():
        if not checkpoint_dir.exists():
            return
        for path in sorted(checkpoint_dir.glob('epoch_*.pt')):
            if path.name not in archived:
                append_jsonl(run / 'checkpoint_hashes.jsonl', {
                    'time': stamp(), 'file': path.name,
                    'bytes': path.stat().st_size, 'sha256': digest(path),
                })
                archived.add(path.name)

    for phase, command in [('training', train), ('evaluation', evaluation)]:
        if stopped:
            break
        if phase == 'evaluation' and not (checkpoint_dir / 'best_val_unseen').exists():
            raise FileNotFoundError('training did not create best_val_unseen')
        log_path = run / f'{phase}.log'
        with log_path.open('w') as log:
            child = subprocess.Popen(
                command, cwd=snapshot, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
            append_jsonl(run / 'events.jsonl', {
                'time': stamp(), 'event': 'started', 'phase': phase, 'pid': child.pid,
            })
            while True:
                exit_code = child.poll()
                status = {
                    'time': stamp(),
                    'phase': phase,
                    'pid': child.pid,
                    'exit_code': exit_code,
                    'log_bytes': log_path.stat().st_size,
                    'log_age_seconds': time.time() - log_path.stat().st_mtime,
                    'disk_free_gib': shutil.disk_usage(run).free / 2**30,
                    'gpu': command_output([
                        'nvidia-smi',
                        '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits',
                    ]).strip(),
                    'alerts': [],
                }
                if status['log_age_seconds'] > 1800 and exit_code is None:
                    status['alerts'].append('no log update for 30 minutes')
                if status['disk_free_gib'] < 12:
                    status['alerts'].append('less than 12 GiB free disk')
                if exit_code not in (None, 0):
                    status['alerts'].append('process exited with error')
                atomic_json(run / 'status.json', status)
                append_jsonl(run / 'telemetry.jsonl', status)
                if status['alerts']:
                    append_jsonl(run / 'alerts.jsonl', status)
                archive_checkpoints()
                if exit_code is not None:
                    append_jsonl(run / 'events.jsonl', {
                        'time': stamp(), 'event': 'exited',
                        'phase': phase, 'exit_code': exit_code,
                    })
                    if exit_code != 0:
                        raise SystemExit(exit_code if exit_code > 0 else 1)
                    break
                time.sleep(args.interval)

    atomic_json(run / 'status.json', {
        'time': stamp(), 'phase': 'stopped' if stopped else 'complete',
    })


if __name__ == '__main__':
    main()
