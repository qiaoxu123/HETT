"""Compare the original buggy baseline with the corrected seed-0 baseline."""

import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
CONTROL = ROOT / '00-control'
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
ORIGINAL = Path('/home/tenant2/Workspace/hett-crotonyl/runs/hett_baseline_fixed_20260911')
CORRECTED = ROOT / '01-teacher-fix/runs/teacher_fix_full_s0'
PREDECESSOR_UNIT = 'hett-corrected-baseline-full-s0-20260911'
PREDECESSOR_STATUS = CONTROL / 'runs/full_corrected_baseline_queue_s0_20260911/status.json'


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def build_commands(output_dir):
    output_dir = Path(output_dir)
    training = [
        PYTHON, str(CONTROL / 'scripts/analyze_training_dynamics.py'),
        '--run', f'original_buggy:0={ORIGINAL}',
        '--run', f'corrected:0={CORRECTED}',
        '--output-dir', str(output_dir / 'training'),
    ]
    navigation = [
        PYTHON, str(CONTROL / 'scripts/analyze_validation_matrix.py'),
        '--run', f'original_buggy={ORIGINAL}', '--run', f'corrected={CORRECTED}',
        '--pair', 'original_buggy:corrected',
        '--output-dir', str(output_dir / 'navigation'),
    ]
    return {'training': training, 'navigation': navigation}


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
        raise SystemExit('corrected seed-0 baseline did not complete')
    commands = build_commands(args.run_dir)
    write(args.run_dir / 'commands.json', commands)
    for name, command in commands.items():
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'analyzing', 'job': name})
        with (args.run_dir / f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=CONTROL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'failed',
                                                 'job': name, 'exit_code': result.returncode})
            raise SystemExit(result.returncode)
    write(args.run_dir / 'status.json', {
        'time': stamp(), 'phase': 'complete',
        'causal_scope': ('training-signal change is per-episode teacher stage1 indexing; '
                         'both runs disable bidirectional task interaction'),
    })


if __name__ == '__main__':
    main()
