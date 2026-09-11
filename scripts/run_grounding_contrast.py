"""Run same-landmark/different-target contrast after all smoke ablations."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
PREDECESSOR_UNIT = 'hett-hypothesis-ablations-20260911'
PREDECESSOR_STATUS = ROOT / '00-control/runs/hypothesis_ablations_20260911/status.json'
LOCK = ROOT / '.gpu-validation.lock'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    while subprocess.run(['systemctl', '--user', 'is-active', '--quiet', PREDECESSOR_UNIT]).returncode == 0:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'waiting',
                                             'unit': PREDECESSOR_UNIT})
        time.sleep(30)
    predecessor = json.loads(PREDECESSOR_STATUS.read_text())
    if predecessor.get('phase') != 'complete':
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'blocked',
                                             'predecessor': predecessor})
        raise SystemExit('predecessor ablations did not complete')

    output = ROOT / '03-grounding/runs/grounding_hard_contrast_smoke_s0'
    checkpoint = ROOT / '03-grounding/runs/grounding_smoke_s0/checkpoints/best_val_unseen'
    command = [PYTHON, '-u', str(ROOT / '03-grounding/scripts/evaluate_grounding_contrasts.py'),
               '--checkpoint', str(checkpoint), '--output-dir', str(output), '--pairs', '16']
    write(args.run_dir / 'command.json', command)
    lock_stream = LOCK.open('w')
    fcntl.flock(lock_stream, fcntl.LOCK_EX)
    with (args.run_dir / 'contrast.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT / '03-grounding', stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'failed',
                                              'exit_code': result.returncode})
        raise SystemExit(result.returncode)
    write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'complete',
                                         'result': str(output / 'summary.json')})


if __name__ == '__main__':
    main()
