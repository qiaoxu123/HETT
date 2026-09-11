"""Fail unless the complete HETT experiment campaign has authoritative artifacts."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
CONTROL = ROOT / '00-control'
BASELINE = Path('/home/tenant2/Workspace/hett-crotonyl/runs/hett_baseline_fixed_20260911')
SEEDS = (0, 17, 42)
TRAIN_VARIANTS = {
    'corrected': ('01-teacher-fix', 'teacher_fix_full'),
    'grounding': ('03-grounding', 'grounding_full'),
    'hypothesis': ('05-hypotheses', 'hypothesis_full'),
    'bidirectional': ('06-bidir', 'bidir_full'),
    'paper_loss_only': ('07-loss-ablation', 'loss_paper_only_full'),
    'no_progress': ('07-loss-ablation', 'loss_no_progress_full'),
    'neither_auxiliary': ('07-loss-ablation', 'loss_neither_full'),
}
EVAL_VARIANTS = {
    'recovery_off': ('02-recovery', 'recovery_off_full_eval'),
    'recovery_on': ('02-recovery', 'recovery_on_full_eval'),
    'combined': ('04-combined', 'combined_full_eval'),
}
QUEUE_RESULTS = {
    'smoke': CONTROL / 'runs/gpu_validation_queue_20260911/status.json',
    'grounding_ablations': CONTROL / 'runs/post_smoke_ablations_20260911/status.json',
    'hypothesis_ablations': CONTROL / 'runs/hypothesis_ablations_20260911/status.json',
    'grounding_contrast': CONTROL / 'runs/grounding_contrast_queue_20260911/status.json',
    'bidirectional_smoke': CONTROL / 'runs/bidir_validation_queue_20260911/status.json',
    'loss_smoke': CONTROL / 'runs/loss_ablation_queue_20260911/status.json',
    'corrected_seed0': CONTROL / 'runs/full_corrected_baseline_queue_s0_20260911/status.json',
    'full_seed0': CONTROL / 'runs/full_seed0_matrix_20260911/status.json',
    'multiseed': CONTROL / 'runs/multiseed_confirmation_20260911/status.json',
    'final_test': CONTROL / 'runs/frozen_final_test_20260911/status.json',
}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def json_lines(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def run_path(mapping, name, seed):
    worktree, prefix = mapping[name]
    return ROOT / worktree / 'runs' / f'{prefix}_s{seed}'


def add(checks, name, passed, evidence):
    checks.append({'name': name, 'passed': bool(passed), 'evidence': evidence})


def audit():
    checks = []
    baseline_status = read_json(BASELINE / 'status.json')
    baseline_epochs = json_lines(BASELINE / 'checkpoints/epoch_metrics.jsonl')
    add(checks, 'original baseline complete', baseline_status is not None and
        baseline_status.get('phase') == 'complete', baseline_status)
    add(checks, 'original baseline has 20 epochs', len(baseline_epochs) == 20,
        {'epochs': len(baseline_epochs)})
    for split in ('val_seen', 'val_unseen'):
        path = BASELINE / 'evaluation' / f'{split}_predictions.pt'
        add(checks, f'original baseline {split} predictions', path.is_file(), str(path))
    baseline_test = BASELINE / 'evaluation/test_unseen_predictions.pt'
    add(checks, 'original baseline did not evaluate test before freeze',
        not baseline_test.exists(), str(baseline_test))

    for name, path in QUEUE_RESULTS.items():
        status = read_json(path)
        accepted = {'complete'} if name != 'final_test' else {'complete', 'auditing'}
        add(checks, f'queue {name} complete', status is not None and
            status.get('phase') in accepted, status)

    development_runs = []
    for name in TRAIN_VARIANTS:
        for seed in SEEDS:
            path = run_path(TRAIN_VARIANTS, name, seed)
            development_runs.append(path)
            status = read_json(path / 'status.json')
            epochs = json_lines(path / 'checkpoints/epoch_metrics.jsonl')
            add(checks, f'{name} seed {seed} complete', status is not None and
                status.get('phase') == 'complete', status)
            add(checks, f'{name} seed {seed} has 20 epochs', len(epochs) == 20,
                {'epochs': len(epochs), 'path': str(path)})
            for artifact in ('provenance.json', 'commands.json', 'checkpoint_hashes.jsonl'):
                target = path / artifact
                add(checks, f'{name} seed {seed} {artifact}', target.is_file(), str(target))
            for split in ('val_seen', 'val_unseen'):
                target = path / 'evaluation' / f'{split}_predictions.pt'
                add(checks, f'{name} seed {seed} {split}', target.is_file(), str(target))
    for name in EVAL_VARIANTS:
        for seed in SEEDS:
            path = run_path(EVAL_VARIANTS, name, seed)
            development_runs.append(path)
            status = read_json(path / 'status.json')
            add(checks, f'{name} seed {seed} complete', status is not None and
                status.get('phase') == 'complete', status)
            for split in ('val_seen', 'val_unseen'):
                target = path / 'evaluation' / f'{split}_predictions.pt'
                add(checks, f'{name} seed {seed} {split}', target.is_file(), str(target))

    seed0_controls = [
        ROOT / '03-grounding/runs/grounding_disabled_full_eval_s0',
        ROOT / '05-hypotheses/runs/hypothesis_disabled_full_eval_s0',
    ]
    development_runs.extend(seed0_controls)
    for path in seed0_controls:
        status = read_json(path / 'status.json')
        add(checks, f'{path.name} complete', status is not None and
            status.get('phase') == 'complete', status)

    leaked = [str(path / 'evaluation/test_unseen_predictions.pt')
              for path in development_runs
              if (path / 'evaluation/test_unseen_predictions.pt').exists()]
    add(checks, 'no declared development run touched test_unseen', not leaked, leaked)

    required_analysis = [
        CONTROL / 'runs/multiseed_confirmation_20260911/analysis/summary.json',
        CONTROL / 'runs/multiseed_confirmation_20260911/analysis/multiseed_metrics.png',
        CONTROL / 'runs/multiseed_confirmation_20260911/analysis/multiseed_paired_deltas.png',
        CONTROL / 'runs/multiseed_confirmation_20260911/training_analysis/summary.json',
        CONTROL / 'runs/multiseed_confirmation_20260911/training_analysis/training_curves.png',
        CONTROL / 'runs/multiseed_confirmation_20260911/training_analysis/weighted_loss_components.png',
    ]
    for path in required_analysis:
        add(checks, f'analysis artifact {path.name}', path.is_file() and path.stat().st_size > 0
            if path.exists() else False, str(path))

    contrast = ROOT / '03-grounding/runs/grounding_hard_contrast_smoke_s0/summary.json'
    add(checks, 'hard contrast summary', contrast.is_file(), str(contrast))
    bidir_log = CONTROL / 'runs/bidir_validation_queue_20260911/gradient_check.log'
    add(checks, 'real GPU bidirectional gradient check', bidir_log.is_file() and
        'GPU_CHECK' in bidir_log.read_text() if bidir_log.exists() else False, str(bidir_log))

    final_dir = CONTROL / 'runs/frozen_final_test_20260911'
    freeze = read_json(final_dir / 'freeze.json')
    add(checks, 'candidate frozen from validation only', freeze is not None and
        freeze.get('selection_split') == 'val_unseen' and
        freeze.get('test_metrics_observed') is False, freeze)
    if freeze:
        selected = freeze['selected']
        final_variants = ('corrected',) if selected == 'corrected' else ('corrected', selected)
        final_outputs = []
        for name in final_variants:
            worktree = ({**TRAIN_VARIANTS, **EVAL_VARIANTS})[name][0]
            for seed in SEEDS:
                run = ROOT / worktree / 'runs' / f'final_test_{name}_s{seed}'
                target = run / 'evaluation/test_unseen_predictions.pt'
                final_outputs.append(target)
                add(checks, f'final test {name} seed {seed}', target.is_file(), str(target))
        if final_outputs:
            freeze_time = (final_dir / 'freeze.json').stat().st_mtime_ns
            add(checks, 'freeze predates every final test output', all(
                path.exists() and path.stat().st_mtime_ns > freeze_time for path in final_outputs),
                {'freeze_mtime_ns': freeze_time})
        final_summary = final_dir / 'analysis/summary.json'
        add(checks, 'final test analysis', final_summary.is_file(), str(final_summary))

    failed = [check for check in checks if not check['passed']]
    return {'complete': not failed, 'checks': checks, 'failed_count': len(failed),
            'passed_count': len(checks) - len(failed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'complete': result['complete'], 'passed': result['passed_count'],
                      'failed': result['failed_count'], 'output': str(args.output)}))
    if not result['complete']:
        sys.exit(1)


if __name__ == '__main__':
    main()
