"""Audit that isolated experiments share the declared HETT training protocol."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'
WORKTREES = ('01-teacher-fix', '02-recovery', '03-grounding',
             '04-combined', '05-hypotheses', '06-bidir',
             '07-loss-ablation')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shell_options(path):
    text = path.read_text()
    # Only parse the actual command. Examples below the "$@" passthrough are comments.
    text = text.split('"$@"', 1)[0]
    return dict(re.findall(r'--([a-zA-Z0-9_]+)\s+([^\\\s"\']+)', text))


def parser_values(worktree):
    code = """
import json, sys
from multiagent.parser import parse_args
sys.argv = ['audit', '--mode', 'train']
a = parse_args()
names = ['direction_loss_weight', 'goal_loss_weight', 'progress_loss_weight',
         'target_loss_weight', 'weight_decay', 'encoder_layers', 'encoder_heads',
         'demb', 'success_dist']
print(json.dumps({name: getattr(a, name) for name in names}))
"""
    output = subprocess.check_output([PYTHON, '-c', code], cwd=ROOT / worktree, text=True)
    return json.loads(output.strip().splitlines()[-1])


def normalized(options):
    converters = {
        'world_size': int, 'seed': int, 'altitude': float, 'learning_rate': float,
        'batch_size': int, 'grad_accum': int, 'epochs': int, 'move_iteration': int,
        'max_action_len': int, 'grid_size': int, 'optim': str, 'feedback': str,
        'train_trajectory_type': str,
    }
    return {name: converters[name](options[name]) for name in converters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    variants = {}
    for worktree in WORKTREES:
        train_path = ROOT / worktree / 'multiagent/train.sh'
        train = normalized(shell_options(train_path))
        defaults = parser_values(worktree)
        variants[worktree] = {
            'train': train,
            'loss_and_model_defaults': defaults,
            'train_script_sha256': sha256(train_path),
        }

    reference = variants[WORKTREES[0]]
    checks = {
        'all_train_scripts_identical': len({row['train_script_sha256'] for row in variants.values()}) == 1,
        'all_effective_protocols_identical': all(
            row['train'] == reference['train'] and
            row['loss_and_model_defaults'] == reference['loss_and_model_defaults']
            for row in variants.values()),
        'paper_core_parameters_match': (
            reference['train']['epochs'] == 20 and
            reference['train']['batch_size'] == 2 and
            reference['train']['learning_rate'] == 1e-4 and
            reference['train']['optim'].lower() == 'adamw' and
            reference['train']['grid_size'] == 5 and
            reference['loss_and_model_defaults']['goal_loss_weight'] == 2.0 and
            reference['loss_and_model_defaults']['direction_loss_weight'] == 1.5 and
            reference['loss_and_model_defaults']['progress_loss_weight'] == 0.1),
        'single_gpu_effective_episode_batch_is_eight': (
            reference['train']['world_size'] == 1 and
            reference['train']['batch_size'] * reference['train']['grad_accum'] == 8),
        'released_auxiliary_target_loss_is_explicit': (
            reference['loss_and_model_defaults']['target_loss_weight'] == 0.1),
        'dataset_and_rollout_protocol_match': (
            reference['train']['train_trajectory_type'] == 'mturk' and
            reference['train']['feedback'] == 'student' and
            reference['train']['max_action_len'] == 20 and
            reference['train']['move_iteration'] == 10),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    report = {
        'paper_source': 'https://arxiv.org/html/2512.14222v1#Sx4.SSx4',
        'paper_declared': {
            'hardware': '4 x RTX A5000 24GB', 'epochs': 20, 'batch_size': 2,
            'learning_rate': 1e-4, 'optimizer': 'AdamW', 'grid_size': 5,
            'loss_weights': {'goal': 2.0, 'direction': 1.5, 'progress': 0.1},
        },
        'single_gpu_adaptation': {
            'world_size': 1, 'micro_batch': 2, 'gradient_accumulation': 4,
            'effective_episode_batch': 8,
            'caveat': ('The paper says four GPUs and batch size 2 but does not explicitly say whether '
                       'batch size is per GPU. The implementation treats it as per-process; accumulation '
                       'therefore preserves an effective batch of eight, not bitwise DDP equivalence.'),
        },
        'released_code_extra': {
            'target_grid_loss_weight': 0.1,
            'paper_status': 'not included among the three losses in Eq. 13 / implementation details',
        },
        'checks': checks,
        'variants': variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
