"""Freeze one variant from validation, then evaluate test_unseen exactly once."""

import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
PREDECESSOR_UNIT = 'hett-multiseed-confirmation-20260911'
PREDECESSOR_STATUS = ROOT / '00-control/runs/multiseed_confirmation_20260911/status.json'
MULTISEED_SUMMARY = ROOT / '00-control/runs/multiseed_confirmation_20260911/analysis/summary.json'
SEEDS = (0, 17, 42)
CANDIDATES = (
    'recovery_on', 'grounding', 'combined', 'hypothesis', 'bidirectional',
    'paper_loss_only', 'no_progress', 'neither_auxiliary',
)
MIN_SR_GAIN_PP = 3.0
MAX_SPL_DROP_PP = 1.0


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def freeze_candidate(summary):
    for name, splits in summary['runs'].items():
        if any(split.startswith('test') for split in splits):
            raise ValueError(f'test result present before freezing: {name}')
    baseline = summary['runs']['corrected']['val_unseen']
    baseline_sr = baseline['sr']['mean']
    baseline_spl = baseline['spl']['mean']
    considered = {}
    eligible = []
    for name in CANDIDATES:
        values = summary['runs'][name]['val_unseen']
        sr = values['sr']['mean']
        spl = values['spl']['mean']
        row = {'sr_mean': sr, 'spl_mean': spl, 'sr_gain_pp': sr - baseline_sr,
               'spl_change_pp': spl - baseline_spl}
        row['eligible'] = (row['sr_gain_pp'] >= MIN_SR_GAIN_PP and
                           row['spl_change_pp'] >= -MAX_SPL_DROP_PP)
        considered[name] = row
        if row['eligible']:
            eligible.append(name)
    selected = max(eligible, key=lambda name: (
        considered[name]['sr_mean'], considered[name]['spl_mean'], name)) if eligible else 'corrected'
    return {
        'selection_split': 'val_unseen', 'selected': selected,
        'rule': {'min_sr_gain_pp': MIN_SR_GAIN_PP, 'max_spl_drop_pp': MAX_SPL_DROP_PP,
                 'tie_break': 'highest mean SR, then mean SPL, then stable name'},
        'baseline': {'sr_mean': baseline_sr, 'spl_mean': baseline_spl},
        'considered': considered,
        'test_metrics_observed': False,
    }


def variant_spec(name, seed):
    baseline = ROOT / f'01-teacher-fix/runs/teacher_fix_full_s{seed}/checkpoints/best_val_unseen'
    specs = {
        'corrected': ('01-teacher-fix', baseline, ['--disable_task_interaction']),
        'recovery_on': ('02-recovery', baseline,
                        ['--disable_task_interaction', '--enable_stage_recovery']),
        'grounding': ('03-grounding',
                      ROOT / f'03-grounding/runs/grounding_full_s{seed}/checkpoints/best_val_unseen',
                      ['--disable_task_interaction', '--enable_region_grounding']),
        'combined': ('04-combined',
                     ROOT / f'03-grounding/runs/grounding_full_s{seed}/checkpoints/best_val_unseen',
                     ['--disable_task_interaction', '--enable_region_grounding',
                      '--enable_stage_recovery']),
        'hypothesis': ('05-hypotheses',
                       ROOT / f'05-hypotheses/runs/hypothesis_full_s{seed}/checkpoints/best_val_unseen',
                       ['--disable_task_interaction', '--enable_multi_hypothesis']),
        'bidirectional': ('06-bidir',
                          ROOT / f'06-bidir/runs/bidir_full_s{seed}/checkpoints/best_val_unseen', []),
        'paper_loss_only': ('07-loss-ablation',
                            ROOT / f'07-loss-ablation/runs/loss_paper_only_full_s{seed}/checkpoints/best_val_unseen',
                            ['--disable_task_interaction', '--target_loss_weight', '0']),
        'no_progress': ('07-loss-ablation',
                        ROOT / f'07-loss-ablation/runs/loss_no_progress_full_s{seed}/checkpoints/best_val_unseen',
                        ['--disable_task_interaction', '--progress_loss_weight', '0']),
        'neither_auxiliary': ('07-loss-ablation',
                              ROOT / f'07-loss-ablation/runs/loss_neither_full_s{seed}/checkpoints/best_val_unseen',
                              ['--disable_task_interaction', '--progress_loss_weight', '0',
                               '--target_loss_weight', '0']),
    }
    return specs[name]


def evaluation_command(name, seed):
    worktree, checkpoint, variant_args = variant_spec(name, seed)
    directory = ROOT / worktree
    run = directory / 'runs' / f'final_test_{name}_s{seed}'
    command = [PYTHON, '-u', str(directory / 'scripts/supervise_experiment.py'),
               '--python', PYTHON, '--run-dir', str(run), '--phase', 'eval',
               '--checkpoint', str(checkpoint), '--epochs', '20', '--max-episodes', '0',
               '--save-every', '20', '--seed', str(seed), '--interval', '60',
               '--variant-arg=--include_test_unseen']
    command.extend(f'--variant-arg={value}' for value in variant_args)
    return command


def final_analysis_command(selected, output_dir):
    command = [PYTHON, str(ROOT / '00-control/scripts/analyze_multiseed.py')]
    names = ('corrected',) if selected == 'corrected' else ('corrected', selected)
    for name in names:
        worktree = variant_spec(name, 0)[0]
        for seed in SEEDS:
            path = ROOT / worktree / 'runs' / f'final_test_{name}_s{seed}'
            command.extend(['--run', f'{name}:{seed}={path}'])
    if selected != 'corrected':
        command.extend(['--pair', f'corrected:{selected}'])
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
        raise SystemExit('multi-seed confirmation did not complete')

    summary = json.loads(MULTISEED_SUMMARY.read_text())
    frozen = freeze_candidate(summary)
    frozen['time'] = stamp()
    frozen['source_summary'] = str(MULTISEED_SUMMARY)
    write(args.run_dir / 'freeze.json', frozen)
    selected = frozen['selected']
    jobs = [('corrected', seed, evaluation_command('corrected', seed)) for seed in SEEDS]
    if selected != 'corrected':
        jobs.extend((selected, seed, evaluation_command(selected, seed)) for seed in SEEDS)
    write(args.run_dir / 'jobs.json', [
        {'variant': name, 'seed': seed, 'command': command} for name, seed, command in jobs])
    for name, seed, command in jobs:
        job = f'{name}_s{seed}'
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'testing',
                                             'job': job, 'frozen_variant': selected})
        append(args.run_dir / 'events.jsonl', {'time': stamp(), 'event': 'started', 'job': job})
        with (args.run_dir / f'{job}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT / '00-control', stdout=log,
                                    stderr=subprocess.STDOUT)
        append(args.run_dir / 'events.jsonl', {'time': stamp(), 'event': 'exited',
                                               'job': job, 'exit_code': result.returncode})
        if result.returncode:
            write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'failed',
                                                 'job': job, 'exit_code': result.returncode})
            raise SystemExit(result.returncode)
    analysis = final_analysis_command(selected, args.run_dir / 'analysis')
    write(args.run_dir / 'analysis_command.json', analysis)
    with (args.run_dir / 'analysis.log').open('w') as log:
        result = subprocess.run(analysis, cwd=ROOT / '00-control', stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'failed',
                                             'job': 'analysis', 'exit_code': result.returncode})
        raise SystemExit(result.returncode)
    write(args.run_dir / 'status.json', {'time': stamp(), 'phase': 'complete',
                                         'frozen_variant': selected,
                                         'analysis': str(args.run_dir / 'analysis')})


if __name__ == '__main__':
    main()
