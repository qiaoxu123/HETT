"""Monitor the complete HETT service chain without acquiring the GPU lock."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time


PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
CONTROL = Path('/home/tenant2/Workspace/hett-experiments/00-control')
ROOT = CONTROL.parent
BASELINE = Path('/home/tenant2/Workspace/hett-crotonyl/runs/hett_baseline_fixed_20260911')
UNITS = (
    'hett-baseline-20260911.service',
    'hett-validation-queue-20260911.service',
    'hett-post-smoke-ablations-20260911.service',
    'hett-hypothesis-ablations-20260911.service',
    'hett-grounding-contrast-20260911.service',
    'hett-bidir-validation-20260911.service',
    'hett-loss-ablation-20260911.service',
    'hett-corrected-baseline-full-s0-20260911.service',
    'hett-full-seed0-matrix-20260911.service',
    'hett-multiseed-confirmation-20260911.service',
)
RUN_STATUS = {
    'hett-baseline-20260911.service': BASELINE / 'status.json',
    'hett-validation-queue-20260911.service': CONTROL / 'runs/gpu_validation_queue_20260911/status.json',
    'hett-post-smoke-ablations-20260911.service': CONTROL / 'runs/post_smoke_ablations_20260911/status.json',
    'hett-hypothesis-ablations-20260911.service': CONTROL / 'runs/hypothesis_ablations_20260911/status.json',
    'hett-grounding-contrast-20260911.service': CONTROL / 'runs/grounding_contrast_queue_20260911/status.json',
    'hett-bidir-validation-20260911.service': CONTROL / 'runs/bidir_validation_queue_20260911/status.json',
    'hett-loss-ablation-20260911.service': CONTROL / 'runs/loss_ablation_queue_20260911/status.json',
    'hett-corrected-baseline-full-s0-20260911.service': CONTROL / 'runs/full_corrected_baseline_queue_s0_20260911/status.json',
    'hett-full-seed0-matrix-20260911.service': CONTROL / 'runs/full_seed0_matrix_20260911/status.json',
    'hett-multiseed-confirmation-20260911.service': CONTROL / 'runs/multiseed_confirmation_20260911/status.json',
}
EXPERIMENT_STATUS = {
    'teacher_fix_smoke': ROOT / '01-teacher-fix/runs/teacher_fix_smoke_s0/status.json',
    'recovery_off_smoke': ROOT / '02-recovery/runs/recovery_off_smoke_s0/status.json',
    'recovery_on_smoke': ROOT / '02-recovery/runs/recovery_on_smoke_s0/status.json',
    'grounding_smoke': ROOT / '03-grounding/runs/grounding_smoke_s0/status.json',
    'grounding_off_smoke': ROOT / '03-grounding/runs/grounding_off_smoke_s0/status.json',
    'grounding_shuffle_language_smoke': ROOT / '03-grounding/runs/grounding_shuffle_language_smoke_s0/status.json',
    'grounding_shuffle_visual_smoke': ROOT / '03-grounding/runs/grounding_shuffle_visual_smoke_s0/status.json',
    'combined_smoke': ROOT / '04-combined/runs/combined_smoke_s0/status.json',
    'multi_hypothesis_smoke': ROOT / '05-hypotheses/runs/multi_hypothesis_smoke_s0/status.json',
    'hypothesis_off_smoke': ROOT / '05-hypotheses/runs/hypothesis_off_smoke_s0/status.json',
    'hypothesis_no_temporal_smoke': ROOT / '05-hypotheses/runs/hypothesis_no_temporal_smoke_s0/status.json',
    'bidir_smoke': ROOT / '06-bidir/runs/bidir_smoke_s0/status.json',
    'loss_released_smoke': ROOT / '07-loss-ablation/runs/loss_released_smoke_s0/status.json',
    'loss_paper_only_smoke': ROOT / '07-loss-ablation/runs/loss_paper_only_smoke_s0/status.json',
    'loss_no_progress_smoke': ROOT / '07-loss-ablation/runs/loss_no_progress_smoke_s0/status.json',
    'loss_neither_smoke': ROOT / '07-loss-ablation/runs/loss_neither_smoke_s0/status.json',
    'teacher_fix_full_s0': ROOT / '01-teacher-fix/runs/teacher_fix_full_s0/status.json',
    'recovery_off_full_eval_s0': ROOT / '02-recovery/runs/recovery_off_full_eval_s0/status.json',
    'recovery_on_full_eval_s0': ROOT / '02-recovery/runs/recovery_on_full_eval_s0/status.json',
    'grounding_disabled_full_eval_s0': ROOT / '03-grounding/runs/grounding_disabled_full_eval_s0/status.json',
    'grounding_full_s0': ROOT / '03-grounding/runs/grounding_full_s0/status.json',
    'combined_full_eval_s0': ROOT / '04-combined/runs/combined_full_eval_s0/status.json',
    'hypothesis_disabled_full_eval_s0': ROOT / '05-hypotheses/runs/hypothesis_disabled_full_eval_s0/status.json',
    'hypothesis_full_s0': ROOT / '05-hypotheses/runs/hypothesis_full_s0/status.json',
    'bidir_full_s0': ROOT / '06-bidir/runs/bidir_full_s0/status.json',
    'loss_paper_only_full_s0': ROOT / '07-loss-ablation/runs/loss_paper_only_full_s0/status.json',
    'loss_no_progress_full_s0': ROOT / '07-loss-ablation/runs/loss_no_progress_full_s0/status.json',
    'loss_neither_full_s0': ROOT / '07-loss-ablation/runs/loss_neither_full_s0/status.json',
}
for _seed in (17, 42):
    EXPERIMENT_STATUS.update({
        f'teacher_fix_full_s{_seed}': ROOT / f'01-teacher-fix/runs/teacher_fix_full_s{_seed}/status.json',
        f'recovery_off_full_eval_s{_seed}': ROOT / f'02-recovery/runs/recovery_off_full_eval_s{_seed}/status.json',
        f'recovery_on_full_eval_s{_seed}': ROOT / f'02-recovery/runs/recovery_on_full_eval_s{_seed}/status.json',
        f'grounding_full_s{_seed}': ROOT / f'03-grounding/runs/grounding_full_s{_seed}/status.json',
        f'combined_full_eval_s{_seed}': ROOT / f'04-combined/runs/combined_full_eval_s{_seed}/status.json',
        f'hypothesis_full_s{_seed}': ROOT / f'05-hypotheses/runs/hypothesis_full_s{_seed}/status.json',
        f'bidir_full_s{_seed}': ROOT / f'06-bidir/runs/bidir_full_s{_seed}/status.json',
        f'loss_paper_only_full_s{_seed}': ROOT / f'07-loss-ablation/runs/loss_paper_only_full_s{_seed}/status.json',
        f'loss_no_progress_full_s{_seed}': ROOT / f'07-loss-ablation/runs/loss_no_progress_full_s{_seed}/status.json',
        f'loss_neither_full_s{_seed}': ROOT / f'07-loss-ablation/runs/loss_neither_full_s{_seed}/status.json',
    })


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def json_lines(path):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def unit_state(unit):
    result = subprocess.run(
        ['systemctl', '--user', 'show', unit, '--property=ActiveState,SubState,MainPID,ExecMainStatus'],
        capture_output=True, text=True,
    )
    values = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key] = value
    return values or {'ActiveState': 'not-found', 'SubState': 'not-found',
                      'MainPID': '0', 'ExecMainStatus': str(result.returncode)}


def snapshot():
    baseline_status = read_json(BASELINE / 'status.json') or {}
    epochs = json_lines(BASELINE / 'checkpoints/epoch_metrics.jsonl')
    batches = json_lines(BASELINE / 'checkpoints/batch_metrics.jsonl')
    units = {unit: unit_state(unit) for unit in UNITS}
    run_statuses = {unit: read_json(path) for unit, path in RUN_STATUS.items()}
    experiments = {name: read_json(path) for name, path in EXPERIMENT_STATUS.items()}
    alerts = list(baseline_status.get('alerts', []))
    for unit, state in units.items():
        if state.get('ActiveState') == 'failed' or (
                state.get('ActiveState') == 'inactive' and state.get('ExecMainStatus') not in ('0', '')):
            alerts.append(f'{unit}: {state}')
        if state.get('ActiveState') in ('inactive', 'not-found'):
            result = run_statuses[unit]
            if not result or result.get('phase') != 'complete':
                alerts.append(f'{unit}: stopped without phase=complete result: {result}')
    for name, status in experiments.items():
        if not status:
            continue
        for alert in status.get('alerts', []):
            alerts.append(f'{name}: {alert}')
        if status.get('phase') in ('failed', 'blocked', 'stopped'):
            detail = {key: status[key] for key in ('exit_code', 'reason') if key in status}
            alerts.append(f'{name}: terminal phase={status.get("phase")}: {detail}')
    return {
        'time': stamp(), 'units': units, 'run_statuses': run_statuses,
        'experiments': experiments,
        'baseline_status': baseline_status,
        'baseline_completed_epochs': len(epochs),
        'baseline_latest_epoch': epochs[-1] if epochs else None,
        'baseline_latest_batch': batches[-1] if batches else None,
        'alerts': alerts,
    }


def compact_run_statuses(statuses):
    tracked = ('phase', 'unit', 'job', 'experiment', 'exit_code', 'reason')
    return {
        unit: ({key: status[key] for key in tracked if key in status}
               if isinstance(status, dict) else status)
        for unit, status in statuses.items()
    }


def signature(value):
    return json.dumps({
        'units': value['units'],
        'run_statuses': compact_run_statuses(value['run_statuses']),
        'experiments': compact_run_statuses(value['experiments']),
        'baseline_phase': value['baseline_status'].get('phase'),
        'baseline_completed_epochs': value['baseline_completed_epochs'],
        'alerts': value['alerts'],
    }, sort_keys=True, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if args.interval < 10:
        parser.error('interval must be at least 10 seconds')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stopping = False

    def stop(*unused):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous_signature = None
    plotted_epochs = -1
    while not stopping:
        current = snapshot()
        temporary = args.output_dir / 'status.json.tmp'
        temporary.write_text(json.dumps(current, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(args.output_dir / 'status.json')
        current_signature = signature(current)
        changed = current_signature != previous_signature
        if changed:
            with (args.output_dir / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(current, ensure_ascii=False) + '\n')
            previous_signature = current_signature
        if changed and current['alerts']:
            with (args.output_dir / 'alerts.jsonl').open('a') as stream:
                stream.write(json.dumps(current, ensure_ascii=False) + '\n')
        if current['baseline_completed_epochs'] != plotted_epochs:
            report = [PYTHON, str(CONTROL / 'scripts/report_live_baseline.py'),
                      '--run-dir', str(BASELINE),
                      '--output-dir', str(args.output_dir / 'baseline_report')]
            subprocess.run(report, cwd=CONTROL, check=True)
            plotted_epochs = current['baseline_completed_epochs']
        time.sleep(args.interval)
    final = snapshot()
    final['monitor_phase'] = 'stopped'
    (args.output_dir / 'status.json').write_text(json.dumps(final, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
