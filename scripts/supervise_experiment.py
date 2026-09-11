"""Run one isolated train/eval experiment and retain reproducibility evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import fcntl


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command_output(command, cwd=None):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60).stdout


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def build_commands(snapshot, run, args):
    checkpoint_dir = run / 'checkpoints'
    common = ['--seed', str(args.seed), '--max_episodes', str(args.max_episodes),
              '--output_dir', str(checkpoint_dir)] + list(args.variant_arg)
    train = ['bash', str(snapshot / 'multiagent/train.sh'), '--epochs', str(args.epochs),
             '--save_every', str(args.save_every)] + common
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else checkpoint_dir / 'best_val_unseen'
    evaluation = ['bash', str(snapshot / 'multiagent/eval.sh'), '--checkpoint', str(checkpoint),
                  '--seed', str(args.seed), '--max_episodes', str(args.max_episodes),
                  '--output_dir', str(run / 'evaluation')] + list(args.variant_arg)
    return train, evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--max-episodes', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--phase', choices=['train-eval', 'eval'], default='train-eval')
    parser.add_argument('--checkpoint')
    parser.add_argument('--variant-arg', action='append', default=[])
    parser.add_argument('--wait-for-unit')
    parser.add_argument('--require-status',
                        help='JSON status that must report phase=complete after waiting')
    parser.add_argument('--lock-file', default='/home/tenant2/Workspace/hett-experiments/.gpu-validation.lock')
    args = parser.parse_args()
    if args.phase == 'eval' and not args.checkpoint:
        parser.error('--checkpoint is required for eval-only runs')
    if args.epochs < 1 or args.save_every < 1 or args.max_episodes < 0:
        parser.error('epochs/save-every must be positive and max-episodes nonnegative')
    root = Path(__file__).resolve().parent.parent
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / 'source'
    snapshot.mkdir()
    for directory in ['multiagent', 'gsamllavanav', 'vlnce', 'scripts']:
        shutil.copytree(root / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns('checkpoints', '__pycache__', '*.pyc'))
    for name in ['README.md', 'SINGLE_GPU.md', 'requirements.txt', 'CHANGELOG.md']:
        if (root / name).exists():
            shutil.copy2(root / name, snapshot / name)
    for name in ['data', 'weights']:
        (snapshot / name).symlink_to((root / name).resolve(), target_is_directory=True)
    (run / 'git.diff').write_text(command_output(['git', 'diff', 'HEAD'], root))
    (run / 'git_status.txt').write_text(command_output(['git', 'status', '--short'], root))
    (run / 'pip_freeze.txt').write_text(command_output([args.python, '-m', 'pip', 'freeze']))
    (run / 'gpu_initial.txt').write_text(command_output(['nvidia-smi']))
    source_hashes = {str(p.relative_to(snapshot)): digest(p)
                     for p in snapshot.rglob('*.py') if 'data' not in p.relative_to(snapshot).parts}
    data_files = [root / 'data/cityrefer/objects.json', root / 'data/cityrefer/processed_descriptions.json']
    data_files += sorted((root / 'data/processed_citynav').glob('citynav_*.json'))
    data_files += [root / 'weights/best.pt', root / 'weights/yolo_v3.cfg']
    inputs = {str(p): dict(resolved=str(p.resolve()), sha256=digest(p), bytes=p.stat().st_size)
              for p in data_files}
    # Large rasters: retain filenames, sizes and mtimes without an expensive full reread.
    rasters = [dict(path=str(p.resolve()), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
               for p in sorted((root / 'data/rgbd').iterdir()) if p.is_file()]
    atomic_json(run / 'provenance.json', dict(started=stamp(), git_head=command_output(['git','rev-parse','HEAD'],root).strip(),
                source_sha256=source_hashes, input_sha256=inputs, raster_inventory=rasters,
                phase=args.phase, seed=args.seed, variant_args=args.variant_arg,
                note='isolated experiment; compare only against explicitly recorded parent run'))
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', PYTHON=args.python,
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    checkpoint_dir = run / 'checkpoints'
    train, evaluation = build_commands(snapshot, run, args)
    atomic_json(run / 'commands.json', dict(train=train, evaluation=evaluation,
                environment_overrides={k:environment[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONUNBUFFERED','PYTHON',
                                                                 'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']}))
    lock_path = Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open('w')
    fcntl.flock(lock_stream, fcntl.LOCK_EX)
    lock_stream.write(f'{os.getpid()} {run}\n')
    lock_stream.flush()

    while args.wait_for_unit and subprocess.run(
            ['systemctl', '--user', 'is-active', '--quiet', args.wait_for_unit]).returncode == 0:
        atomic_json(run / 'status.json', dict(time=stamp(), phase='waiting',
                    waiting_for_unit=args.wait_for_unit))
        time.sleep(args.interval)
    if args.require_status:
        required = json.loads(Path(args.require_status).read_text())
        if required.get('phase') != 'complete':
            atomic_json(run / 'status.json', dict(time=stamp(), phase='blocked',
                        reason='required predecessor did not complete', required_status=required))
            raise SystemExit('required predecessor did not complete successfully')

    stopped = False
    child = None
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    seen = set()
    def archive_hashes():
        if checkpoint_dir.exists():
            for path in sorted(checkpoint_dir.glob('epoch_*.pt')):
                if path.name not in seen:
                    append(run / 'checkpoint_hashes.jsonl', dict(time=stamp(), file=path.name,
                           bytes=path.stat().st_size, sha256=digest(path)))
                    seen.add(path.name)
    phases = [('evaluation', evaluation)] if args.phase == 'eval' else [
        ('training', train), ('evaluation', evaluation)
    ]
    for phase, command in phases:
        if stopped:
            break
        if phase == 'evaluation':
            checkpoint_index = command.index('--checkpoint') + 1
            if not Path(command[checkpoint_index]).exists():
                raise FileNotFoundError(f'evaluation checkpoint missing: {command[checkpoint_index]}')
        with (run / (phase + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT)
            append(run / 'events.jsonl', dict(time=stamp(), event='started', phase=phase, pid=child.pid))
            while True:
                code = child.poll()
                log_path = run / (phase + '.log')
                age = time.time() - log_path.stat().st_mtime
                status = dict(time=stamp(), phase=phase, pid=child.pid, exit_code=code,
                    log_bytes=log_path.stat().st_size, log_age_seconds=age,
                    disk_free_gib=shutil.disk_usage(run).free / 2**30,
                    gpu=command_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                                        '--format=csv,noheader,nounits']).strip(),
                    memory=command_output(['free','-m']).strip())
                status['alerts'] = []
                if age > 1800 and code is None:
                    status['alerts'].append('No log update for 30 minutes')
                if status['disk_free_gib'] < 12:
                    status['alerts'].append('Less than 12 GiB free disk')
                try:
                    gpu_fields = [float(x.strip()) for x in status['gpu'].splitlines()[0].split(',')]
                    if gpu_fields[3] >= 85:
                        status['alerts'].append('GPU temperature at least 85C')
                except (ValueError, IndexError):
                    status['alerts'].append('GPU telemetry unavailable')
                if code not in (None, 0):
                    status['alerts'].append('Process exited with error; inspect phase log')
                append(run / 'telemetry.jsonl', status)
                atomic_json(run / 'status.json', status)
                if status['alerts']:
                    append(run / 'alerts.jsonl', status)
                archive_hashes()
                if code is not None:
                    append(run / 'events.jsonl', dict(time=stamp(), event='exited', phase=phase, exit_code=code))
                    if code != 0:
                        raise SystemExit(code if code > 0 else 1)
                    break
                time.sleep(args.interval)
        if phase == 'training':
            best = checkpoint_dir / 'best_val_unseen'
            append(run / 'checkpoint_hashes.jsonl', dict(time=stamp(), file=best.name,
                   bytes=best.stat().st_size, sha256=digest(best)))
    atomic_json(run / 'status.json', dict(time=stamp(), phase='stopped' if stopped else 'complete'))


if __name__ == '__main__':
    main()
