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
    for directory in ('multiagent', 'scripts', 'tests', 'docs', 'analysis'):
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
    parser.add_argument(
        '--phase',
        choices=(
            'train', 'eval', 'train-eval', 'static-belief',
            'visual-attributes-phase1', 'visual-attributes-handcrafted',
            'visual-attributes-frozen', 'visual-attributes-zeroshot',
            'visual-attributes-finetune', 'visual-attributes-view-ablation',
            'visual-attributes-isolation-ablation', 'visual-attributes-multiview',
            'visual-attributes-candidate-verification',
            'visual-attributes-best-seed',
            'visual-attributes-sam',
            'scene-grounding-template', 'scene-grounding-dataset',
            'scene-grounding-evidence', 'scene-grounding-distance',
            'scene-grounding-oracle', 'scene-grounding-matcher',
            'scene-grounding-hard-negatives', 'scene-grounding-belief-update',
            'geometry-reasoner', 'geometry-visualize',
        ),
        default='train-eval',
    )
    parser.add_argument('--dataset-root', type=Path, help='required for --phase grounding')
    parser.add_argument('--pointcloud-root', type=Path, help='required for --phase pointcloud-viewer')
    parser.add_argument('--feature-file', type=Path)
    parser.add_argument('--attribute-checkpoint', type=Path)
    parser.add_argument('--b0-cache-root', type=Path)
    parser.add_argument('--sam-source', type=Path)
    parser.add_argument('--sam-checkpoint', type=Path)
    parser.add_argument('--metrics-file', type=Path)
    parser.add_argument('--checkpoint')
    parser.add_argument('--pythonpath', help='optional isolated dependency directory')
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
    if args.pythonpath:
        prior_pythonpath = environment.get('PYTHONPATH')
        environment['PYTHONPATH'] = (
            args.pythonpath if not prior_pythonpath
            else f'{args.pythonpath}:{prior_pythonpath}'
        )
    phases = []
    if args.phase == 'static-belief':
        phases.append((
            'static_belief',
            [args.python, 'scripts/train_static_belief.py',
             '--data-root', str((source / 'data').resolve()),
             '--output', str((run / 'artifacts').resolve()), *args.variant_arg],
        ))
    elif args.phase == 'visual-attributes-phase1':
        dataset = (run / 'artifacts' / 'dataset').resolve()
        phases.extend([
            ('attribute_census', [args.python, 'analysis/visual_attribute_census.py',
                                  '--data-root', str((source / 'data').resolve()),
                                  '--output', str((run / 'artifacts' / 'census').resolve())]),
            ('attribute_dataset', [args.python, 'scripts/build_visual_attribute_dataset.py',
                                   '--data-root', str((source / 'data').resolve()), '--output', str(dataset)]),
            ('attribute_handcrafted', [args.python, 'scripts/train_visual_attributes.py',
                                       '--dataset', str(dataset),
                                       '--output', str((run / 'artifacts' / 'handcrafted').resolve()),
                                       *args.variant_arg]),
        ])
    elif args.phase == 'visual-attributes-handcrafted':
        if args.dataset_root is None:
            parser.error('--dataset-root is required for --phase visual-attributes-handcrafted')
        phases.append((
            'attribute_handcrafted',
            [args.python, 'scripts/train_visual_attributes.py',
             '--dataset', str(args.dataset_root.resolve()),
             '--output', str((run / 'artifacts' / 'handcrafted').resolve()),
             *args.variant_arg],
        ))
    elif args.phase == 'visual-attributes-frozen':
        if args.dataset_root is None:
            parser.error('--dataset-root is required for --phase visual-attributes-frozen')
        dataset = args.dataset_root.resolve()
        encoder_batches = [('darknet', 128), ('dinov2', 64), ('siglip', 48), ('siglip2', 24)]
        for encoder, batch_size in encoder_batches:
            feature_file = (run / 'artifacts' / 'features' / f'{encoder}_crop.pt').resolve()
            probe_dir = (run / 'artifacts' / 'probes' / encoder).resolve()
            phases.extend([
                (f'extract_{encoder}', [
                    args.python, 'scripts/extract_visual_attribute_features.py',
                    '--dataset', str(dataset), '--output', str(feature_file),
                    '--encoder', encoder, '--representations', 'crop',
                    '--batch-size', str(batch_size),
                ]),
                (f'probe_{encoder}', [
                    args.python, 'scripts/train_visual_attributes.py',
                    '--dataset', str(dataset), '--output', str(probe_dir),
                    '--feature-file', str(feature_file), '--representations', 'crop',
                    '--epochs', '30', *args.variant_arg,
                ]),
            ])
        for encoder in ('siglip', 'siglip2'):
            model_name = ('google/siglip-base-patch16-224' if encoder == 'siglip'
                          else 'google/siglip2-base-patch16-256')
            phases.append((f'zeroshot_{encoder}', [
                args.python, 'scripts/evaluate_visual_attributes.py',
                '--dataset', str(dataset),
                '--output', str((run / 'artifacts' / 'zeroshot' / encoder).resolve()),
                '--model-name', model_name, '--representation', 'crop',
            ]))
    elif args.phase == 'visual-attributes-zeroshot':
        if args.dataset_root is None:
            parser.error('--dataset-root is required for --phase visual-attributes-zeroshot')
        for encoder, model_name in (
            ('siglip', 'google/siglip-base-patch16-224'),
            ('siglip2', 'google/siglip2-base-patch16-256'),
        ):
            phases.append((f'zeroshot_{encoder}', [
                args.python, 'scripts/evaluate_visual_attributes.py',
                '--dataset', str(args.dataset_root.resolve()),
                '--output', str((run / 'artifacts' / 'zeroshot' / encoder).resolve()),
                '--model-name', model_name, '--representation', 'crop',
            ]))
    elif args.phase == 'visual-attributes-finetune':
        if args.dataset_root is None:
            parser.error('--dataset-root is required for --phase visual-attributes-finetune')
        for encoder, model_name, batch_size in (
            ('siglip', 'google/siglip-base-patch16-224', 32),
            ('siglip2', 'google/siglip2-base-patch16-256', 24),
        ):
            phases.append((f'finetune_{encoder}', [
                args.python, 'scripts/train_visual_attribute_finetune.py',
                '--dataset', str(args.dataset_root.resolve()),
                '--output', str((run / 'artifacts' / encoder).resolve()),
                '--model-name', model_name, '--representation', 'crop',
                '--batch-size', str(batch_size), '--epochs', '3',
                '--unfreeze-blocks', '1', *args.variant_arg,
            ]))
    elif args.phase == 'visual-attributes-view-ablation':
        dataset = (run / 'artifacts' / 'dataset').resolve()
        feature_file = (run / 'artifacts' / 'features' / 'siglip2_crop.pt').resolve()
        phases.extend([
            ('attribute_view_dataset', [
                args.python, 'scripts/build_visual_attribute_dataset.py',
                '--data-root', str((source / 'data').resolve()), '--output', str(dataset),
                '--heights', '20', '40', '80', '140', '--distances', '10', '30', '60', '100',
            ]),
            ('extract_siglip2', [
                args.python, 'scripts/extract_visual_attribute_features.py',
                '--dataset', str(dataset), '--output', str(feature_file), '--encoder', 'siglip2',
                '--representations', 'crop', '--batch-size', '24',
            ]),
            ('probe_siglip2', [
                args.python, 'scripts/train_visual_attributes.py', '--dataset', str(dataset),
                '--output', str((run / 'artifacts' / 'probe').resolve()),
                '--feature-file', str(feature_file), '--representations', 'crop', '--epochs', '30',
            ]),
        ])
    elif args.phase == 'visual-attributes-isolation-ablation':
        if args.dataset_root is None:
            parser.error('--dataset-root is required for --phase visual-attributes-isolation-ablation')
        dataset = args.dataset_root.resolve()
        feature_file = (run / 'artifacts' / 'features' / 'siglip2_all.pt').resolve()
        phases.extend([
            ('extract_siglip2', [
                args.python, 'scripts/extract_visual_attribute_features.py', '--dataset', str(dataset),
                '--output', str(feature_file), '--encoder', 'siglip2',
                '--representations', 'whole', 'crop', 'masked', '--batch-size', '24',
            ]),
            ('probe_siglip2', [
                args.python, 'scripts/train_visual_attributes.py', '--dataset', str(dataset),
                '--output', str((run / 'artifacts' / 'probe').resolve()), '--feature-file', str(feature_file),
                '--representations', 'whole', 'crop', 'masked', '--epochs', '30',
            ]),
        ])
        if args.sam_source is not None and args.sam_checkpoint is not None:
            sam_features = (run / 'artifacts' / 'features' / 'siglip2_sam.pt').resolve()
            phases.extend([
                ('extract_sam_siglip2', [
                    args.python, 'scripts/extract_sam_attribute_features.py', '--dataset', str(dataset),
                    '--output', str(sam_features), '--sam-source', str(args.sam_source.resolve()),
                    '--sam-checkpoint', str(args.sam_checkpoint.resolve()),
                ]),
                ('probe_sam_siglip2', [
                    args.python, 'scripts/train_visual_attributes.py', '--dataset', str(dataset),
                    '--output', str((run / 'artifacts' / 'probe_sam').resolve()),
                    '--feature-file', str(sam_features), '--representations', 'sam_masked', '--epochs', '30',
                ]),
            ])
    elif args.phase == 'visual-attributes-multiview':
        if args.dataset_root is None or args.feature_file is None:
            parser.error('--dataset-root and --feature-file are required for --phase visual-attributes-multiview')
        phases.append(('multiview_probe', [
            args.python, 'scripts/train_visual_attributes.py', '--dataset', str(args.dataset_root.resolve()),
            '--output', str((run / 'artifacts').resolve()), '--feature-file', str(args.feature_file.resolve()),
            '--representations', 'crop', '--epochs', '30', *args.variant_arg,
        ]))
    elif args.phase == 'visual-attributes-candidate-verification':
        if args.attribute_checkpoint is None or args.b0_cache_root is None:
            parser.error('--attribute-checkpoint and --b0-cache-root are required for candidate verification')
        phases.append(('candidate_verification', [
            args.python, 'scripts/evaluate_candidate_verification.py',
            '--data-root', str((source / 'data').resolve()), '--output', str((run / 'artifacts').resolve()),
            '--checkpoint', str(args.attribute_checkpoint.resolve()),
            '--b0-cache-root', str(args.b0_cache_root.resolve()), *args.variant_arg,
        ]))
    elif args.phase == 'visual-attributes-best-seed':
        if args.dataset_root is None:
            parser.error('--dataset-root is required for --phase visual-attributes-best-seed')
        phases.append(('finetune_siglip2', [
            args.python, 'scripts/train_visual_attribute_finetune.py',
            '--dataset', str(args.dataset_root.resolve()), '--output', str((run / 'artifacts').resolve()),
            '--model-name', 'google/siglip2-base-patch16-256', '--representation', 'crop',
            '--batch-size', '24', '--epochs', '3', '--unfreeze-blocks', '1', *args.variant_arg,
        ]))
    elif args.phase == 'visual-attributes-sam':
        if args.dataset_root is None or args.sam_source is None or args.sam_checkpoint is None:
            parser.error('--dataset-root, --sam-source, and --sam-checkpoint are required for --phase visual-attributes-sam')
        feature_file = (run / 'artifacts' / 'features' / 'siglip2_sam.pt').resolve()
        phases.extend([
            ('extract_sam_siglip2', [
                args.python, 'scripts/extract_sam_attribute_features.py', '--dataset', str(args.dataset_root.resolve()),
                '--output', str(feature_file), '--sam-source', str(args.sam_source.resolve()),
                '--sam-checkpoint', str(args.sam_checkpoint.resolve()),
            ]),
            ('probe_sam_siglip2', [
                args.python, 'scripts/train_visual_attributes.py', '--dataset', str(args.dataset_root.resolve()),
                '--output', str((run / 'artifacts' / 'probe').resolve()), '--feature-file', str(feature_file),
                '--representations', 'crop', 'sam_masked', '--epochs', '30',
            ]),
        ])
    elif args.phase == 'scene-grounding-template':
        phases.append(('scene_template_census', [
            args.python, 'scripts/analyze_scene_templates.py',
            '--data-root', str((source / 'data').resolve()),
            '--output', str((run / 'artifacts').resolve()),
        ]))
    elif args.phase == 'scene-grounding-dataset':
        phases.append(('scene_dataset', [
            args.python, 'scripts/build_scene_grounding_dataset.py',
            '--data-root', str((source / 'data').resolve()),
            '--output', str((run / 'artifacts').resolve()), *args.variant_arg,
        ]))
    elif args.phase == 'scene-grounding-evidence':
        if args.dataset_root is None or args.attribute_checkpoint is None:
            parser.error('--dataset-root and --attribute-checkpoint are required for scene evidence')
        phases.append(('scene_evidence', [
            args.python, 'scripts/extract_scene_evidence.py',
            '--dataset', str(args.dataset_root.resolve()),
            '--checkpoint', str(args.attribute_checkpoint.resolve()),
            '--output', str((run / 'artifacts').resolve()), *args.variant_arg,
        ]))
    elif args.phase == 'scene-grounding-distance':
        if args.dataset_root is None or args.feature_file is None:
            parser.error('--dataset-root and --feature-file (evidence directory) are required')
        phases.append(('scene_distance_curve', [
            args.python, 'scripts/evaluate_scene_distance_curve.py',
            '--dataset', str(args.dataset_root.resolve()),
            '--evidence', str(args.feature_file.resolve()),
            '--output', str((run / 'artifacts').resolve()),
        ]))
    elif args.phase == 'scene-grounding-oracle':
        if args.b0_cache_root is None:
            parser.error('--b0-cache-root is required for scene oracle')
        phases.append(('scene_oracle', [
            args.python, 'scripts/evaluate_scene_oracle.py',
            '--data-root', str((source / 'data').resolve()),
            '--b0-cache-root', str(args.b0_cache_root.resolve()),
            '--output', str((run / 'artifacts').resolve()),
        ]))
    elif args.phase == 'scene-grounding-matcher':
        if args.feature_file is None:
            parser.error('--feature-file must point to Phase-4 metrics.json')
        phases.append(('scene_matcher_gate', [args.python, 'scripts/train_scene_matcher.py',
            '--gate-metrics', str(args.feature_file.resolve()), '--output', str((run / 'artifacts').resolve())]))
    elif args.phase == 'scene-grounding-hard-negatives':
        if args.feature_file is None:
            parser.error('--feature-file must point to Phase-4 metrics.json')
        phases.append(('scene_hard_negatives', [args.python, 'scripts/evaluate_scene_hard_negatives.py',
            '--distance-metrics', str(args.feature_file.resolve()), '--output', str((run / 'artifacts').resolve())]))
    elif args.phase == 'scene-grounding-belief-update':
        if args.feature_file is None:
            parser.error('--feature-file must point to Phase-4 metrics.json')
        phases.append(('scene_belief_gate', [args.python, 'scripts/evaluate_scene_belief_update.py',
            '--gate-metrics', str(args.feature_file.resolve()), '--output', str((run / 'artifacts').resolve())]))
    elif args.phase == 'geometry-reasoner':
        if args.b0_cache_root is None: parser.error('--b0-cache-root is required')
        phases.append(('geometry_reasoner', [args.python, 'scripts/evaluate_geometry_reasoner.py',
            '--data-root', str((source / 'data').resolve()), '--b0-cache-root', str(args.b0_cache_root.resolve()),
            '--output', str((run / 'artifacts').resolve())]))
    elif args.phase == 'geometry-visualize':
        if args.b0_cache_root is None or args.metrics_file is None: parser.error('--b0-cache-root and --metrics-file are required')
        phases.append(('geometry_visualize', [args.python, 'scripts/visualize_geometry_reasoner.py',
            '--data-root', str((source / 'data').resolve()), '--b0-cache-root', str(args.b0_cache_root.resolve()),
            '--metrics', str(args.metrics_file.resolve()), '--output', str((run / 'artifacts').resolve()), *args.variant_arg]))
    else:
        if args.phase in ('train', 'train-eval'):
            phases.append(('training', _command(args.python, 'train', args.checkpoint, args.variant_arg)))
        if args.phase in ('eval', 'train-eval'):
            checkpoint = args.checkpoint or checkpoint_dir / 'best_val_unseen'
            phases.append(('evaluation', _command(args.python, 'eval', checkpoint, args.variant_arg)))
    _json(run / 'commands.json', {name: command for name, command in phases})

    cpu_only = args.phase in {
        'scene-grounding-template', 'scene-grounding-dataset',
        'scene-grounding-distance', 'scene-grounding-oracle',
        'scene-grounding-matcher', 'scene-grounding-hard-negatives',
        'scene-grounding-belief-update',
        'geometry-reasoner', 'geometry-visualize',
    }
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
                    cwd=(source if args.phase in (
                        'static-belief', 'visual-attributes-phase1',
                        'visual-attributes-handcrafted', 'visual-attributes-frozen',
                        'visual-attributes-zeroshot', 'visual-attributes-finetune',
                        'visual-attributes-view-ablation', 'visual-attributes-isolation-ablation',
                        'visual-attributes-multiview', 'visual-attributes-candidate-verification',
                        'visual-attributes-best-seed', 'visual-attributes-sam',
                        'scene-grounding-template', 'scene-grounding-dataset',
                        'scene-grounding-evidence', 'scene-grounding-distance',
                        'scene-grounding-oracle', 'scene-grounding-matcher',
                        'scene-grounding-hard-negatives', 'scene-grounding-belief-update',
                        'geometry-reasoner', 'geometry-visualize',
                    ) else source / 'multiagent'),
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
