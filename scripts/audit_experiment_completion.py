"""Fail unless the complete HETT experiment campaign has authoritative artifacts."""

import argparse
import hashlib
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
EXPECTED_HEADS = {
    '01-teacher-fix': '507355bb8d6631ab415a03fd74fd26113c099945',
    '02-recovery': 'b1c55c2a37332290b0afa59de1c66d537a989502',
    '03-grounding': '6630459986044da2144128ed2a2c70dd4fb762b2',
    '04-combined': 'bf132b044e2af8f1f5c76d21d3c4f33c54282c25',
    '05-hypotheses': '55cb3bfaf9ff2f5dc7ee014a58218ce834b32768',
    '06-bidir': 'b76f9b659dc496583ddcb27745f58a09883cb746',
    '07-loss-ablation': '9b50ad4a636f4985c4392f2cc799c9cf44d46907',
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
    'bugfix_analysis': CONTROL / 'runs/bugfix_analysis_20260911/status.json',
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


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


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
    hygiene = read_json(BASELINE / 'evaluation_hygiene.json')
    source = BASELINE / 'source/multiagent/main.py'
    provenance = read_json(BASELINE / 'provenance.json')
    add(checks, 'baseline evaluation hygiene record exists', hygiene is not None, hygiene)
    add(checks, 'baseline evaluation source matches recorded hygiene hash',
        hygiene is not None and source.is_file() and digest(source) == hygiene.get('sha256_after'),
        {'actual': digest(source) if source.is_file() else None,
         'expected': hygiene.get('sha256_after') if hygiene else None})
    original_hash = (provenance or {}).get('source_sha256', {}).get('multiagent/main.py')
    add(checks, 'baseline original training hash preserved in provenance',
        hygiene is not None and original_hash == hygiene.get('sha256_before'),
        {'provenance': original_hash,
         'expected': hygiene.get('sha256_before') if hygiene else None})

    reference_dir = CONTROL / 'runs/reference_baseline_audit_20260911'
    reference = read_json(reference_dir / 'report.json')
    reference_audit = (reference or {}).get('audit', {})
    reference_artifacts = [reference_dir / name
                           for name in ('report.json', 'README.md', 'comparison.png')]
    add(checks, 'historical baseline audit artifacts',
        all(path.is_file() and path.stat().st_size > 0 for path in reference_artifacts),
        [str(path) for path in reference_artifacts])
    add(checks, 'historical baseline comparability is qualified',
        reference is not None and
        reference_audit.get('has_epoch_0_to_11_log') is True and
        reference_audit.get('has_epoch_11_to_19_resume_log') is True and
        reference_audit.get('has_checkpoint') is False and
        reference_audit.get('single_continuous_lineage') is False and
        reference_audit.get('strict_paper_loss_match') is False,
        reference_audit)

    protocol = read_json(CONTROL / 'runs/training_protocol_audit_20260911/report.json')
    protocol_checks = (protocol or {}).get('checks', {})
    protocol_variants = (protocol or {}).get('variants', {})
    add(checks, 'all seven worktrees pass paper protocol audit',
        len(protocol_variants) == 7 and protocol_checks and all(protocol_checks.values()),
        {'variants': sorted(protocol_variants), 'checks': protocol_checks})

    for name, path in QUEUE_RESULTS.items():
        status = read_json(path)
        accepted = {'complete'} if name != 'final_test' else {'complete', 'auditing'}
        add(checks, f'queue {name} complete', status is not None and
            status.get('phase') in accepted, status)

    development_runs = []
    variant_provenance = {}
    for name in TRAIN_VARIANTS:
        variant_provenance[name] = []
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
            variant_provenance[name].append(read_json(path / 'provenance.json'))
            for split in ('val_seen', 'val_unseen'):
                target = path / 'evaluation' / f'{split}_predictions.pt'
                add(checks, f'{name} seed {seed} {split}', target.is_file(), str(target))
    for name in EVAL_VARIANTS:
        variant_provenance[name] = []
        for seed in SEEDS:
            path = run_path(EVAL_VARIANTS, name, seed)
            development_runs.append(path)
            status = read_json(path / 'status.json')
            variant_provenance[name].append(read_json(path / 'provenance.json'))
            add(checks, f'{name} seed {seed} complete', status is not None and
                status.get('phase') == 'complete', status)
            for split in ('val_seen', 'val_unseen'):
                target = path / 'evaluation' / f'{split}_predictions.pt'
                add(checks, f'{name} seed {seed} {split}', target.is_file(), str(target))

    all_variants = {**TRAIN_VARIANTS, **EVAL_VARIANTS}
    for name, (worktree, unused_prefix) in all_variants.items():
        provenances = variant_provenance[name]
        heads = [item.get('git_head') if item else None for item in provenances]
        source_hashes = [item.get('source_sha256') if item else None for item in provenances]
        add(checks, f'{name} uses declared source commit for all seeds',
            len(heads) == len(SEEDS) and all(
                head == EXPECTED_HEADS[worktree] for head in heads),
            {'expected': EXPECTED_HEADS[worktree], 'actual': heads})
        add(checks, f'{name} source snapshot identical across seeds',
            len(source_hashes) == len(SEEDS) and source_hashes[0] is not None and
            all(value == source_hashes[0] for value in source_hashes[1:]),
            {'source_hash_sets': [len(value) if value else None for value in source_hashes],
             'identical': (source_hashes[0] is not None and
                           all(value == source_hashes[0] for value in source_hashes[1:]))})

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
        CONTROL / 'runs/bugfix_analysis_20260911/training/summary.json',
        CONTROL / 'runs/bugfix_analysis_20260911/training/training_curves.png',
        CONTROL / 'runs/bugfix_analysis_20260911/training/weighted_loss_components.png',
        CONTROL / 'runs/bugfix_analysis_20260911/navigation/summary.json',
        CONTROL / 'runs/bugfix_analysis_20260911/navigation/comparison.png',
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

    geometry = Path('/home/tenant2/Workspace/hett-crotonyl/runs/local_geometry_probe_20260911')
    geometry_required = ('REPORT.md', 'summary.json', 'metrics.json', 'comparison.png',
                         'cases.png', 'protocol.json', 'fit_protocol.json')
    missing_geometry = [name for name in geometry_required
                        if not (geometry / name).is_file() or (geometry / name).stat().st_size == 0]
    add(checks, 'local geometry result artifacts', not missing_geometry, missing_geometry)
    data_audit = read_json(geometry / 'data_audit.json')
    add(checks, 'local geometry train/unseen split and finite features',
        data_audit is not None and not data_audit.get('map_overlap') and
        not data_audit.get('ply_length_failures') and
        all(item.get('finite_features') for item in data_audit.get('checks', [])), data_audit)
    perturbation = read_json(geometry / 'gt_perturbation_audit.json')
    add(checks, 'local geometry student control ignores perturbed GT fields',
        perturbation is not None and perturbation.get('episodes', 0) > 0 and
        all(value == 0 for value in perturbation.get('max_trajectory_difference', [])),
        perturbation)
    usage = read_json(geometry / 'geometry_usage_audit.json')
    usage_checks = (usage or {}).get('checks', [])
    add(checks, 'local geometry branch is active and language-sensitive',
        len(usage_checks) == 3 and all(
            item.get('max_geometry_weight_change', 0) > 0 and
            item.get('max_logits_change_zero_geometry', 0) > 0 and
            item.get('positions_changed_by_shuffled_text', 0) > 0
            for item in usage_checks), usage)
    geometry_protocol = read_json(geometry / 'protocol.json')
    fit_protocol = read_json(geometry / 'fit_protocol.json')
    add(checks, 'local geometry excludes world coordinates and semantic labels',
        geometry_protocol is not None and
        geometry_protocol.get('world_coords_input') is False and
        geometry_protocol.get('semantic_labels_input') is False, geometry_protocol)
    add(checks, 'local geometry uses three fixed seeds and last-epoch selection',
        fit_protocol is not None and len(fit_protocol.get('seeds', [])) == 3 and
        fit_protocol.get('selection', '').startswith('last epoch'), fit_protocol)

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
