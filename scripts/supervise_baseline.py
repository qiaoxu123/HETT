"""Run an isolated baseline and retain provenance, telemetry and final evaluation."""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--max-episodes', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--train-fine-altitude', action='store_true')
    parser.add_argument('--fine-altitude-levels', type=float, nargs='+', default=[50., 40.])
    parser.add_argument('--wait-run', default=None,
                        help='snapshot now, then wait for this run to complete before using the GPU')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / 'source'
    snapshot.mkdir()
    for directory in ['multiagent', 'gsamllavanav', 'vlnce', 'scripts']:
        shutil.copytree(root / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns('checkpoints', '__pycache__', '*.pyc'))
    for name in ['README.md', 'SINGLE_GPU.md', 'requirements.txt', 'CHANGELOG.md']:
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
                variant=(
                    'teacher-forced 50-to-low fine-altitude training; task interaction disabled'
                    if args.train_fine_altitude else
                    'fixed training code, task interaction disabled; not historical buggy baseline'
                )))
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', PYTHON=args.python,
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    checkpoint_dir = run / 'checkpoints'
    train = ['bash', str(snapshot / 'multiagent/train.sh'), '--disable_task_interaction',
             '--epochs', str(args.epochs), '--max_episodes', str(args.max_episodes),
             '--batch_size', str(args.batch_size), '--grad_accum', str(args.grad_accum),
             '--output_dir', str(checkpoint_dir)]
    if args.train_fine_altitude:
        train += ['--train_fine_altitude', '--fine_altitude_levels',
                  *[str(level) for level in args.fine_altitude_levels]]
    evaluation = ['bash', str(snapshot / 'multiagent/eval.sh'), '--disable_task_interaction',
                  '--checkpoint', str(checkpoint_dir / 'best_val_unseen'),
                  '--max_episodes', str(args.max_episodes), '--output_dir', str(run / 'evaluation')]
    phases = [('training', train), ('evaluation', evaluation)]
    dynamic_evaluation = None
    if args.train_fine_altitude:
        dynamic_evaluation = [
            'bash', str(snapshot / 'multiagent/eval.sh'), '--disable_task_interaction',
            '--checkpoint', str(checkpoint_dir / 'best_val_unseen'),
            '--max_episodes', str(args.max_episodes),
            '--output_dir', str(run / 'evaluation_dynamic'),
            '--eval_splits', 'val_seen', 'val_unseen',
            '--fine_altitude_refinement', '--fine_altitude_levels',
            *[str(level) for level in args.fine_altitude_levels],
        ]
        phases.append(('evaluation_dynamic', dynamic_evaluation))
    atomic_json(run / 'commands.json', dict(train=train, evaluation=evaluation,
                evaluation_dynamic=dynamic_evaluation,
                environment_overrides={k:environment[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONUNBUFFERED','PYTHON',
                                                                 'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']}))
    if args.wait_run:
        predecessor_status = Path(args.wait_run).resolve() / 'status.json'
        while True:
            try:
                predecessor = json.loads(predecessor_status.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                predecessor = {}
            if predecessor.get('phase') == 'complete':
                break
            exit_code = predecessor.get('exit_code')
            if exit_code not in (None, 0):
                atomic_json(run / 'status.json', dict(
                    time=stamp(), phase='blocked',
                    reason=f'predecessor failed with exit {exit_code}'))
                raise SystemExit(1)
            atomic_json(run / 'status.json', dict(
                time=stamp(), phase='queued', waiting_for=str(predecessor_status)))
            time.sleep(args.interval)
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
    for phase, command in phases:
        if stopped:
            break
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
