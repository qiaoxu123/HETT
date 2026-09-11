"""Prove that queued cross-branch checkpoints have compatible parameter keys."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path('/home/tenant2/Workspace/hett-experiments')
PYTHON = '/home/tenant2/miniconda3/envs/AirVLN39/bin/python'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_keys(worktree, region=False, hypothesis=False, disable_interaction=True):
    code = f"""
import json
from types import SimpleNamespace
from multiagent.models.ET_haa import ET
a=SimpleNamespace(grid_size=5,demb=768,encoder_heads=12,encoder_layers=2,
 dropout_transformer_encoder=.1,num_input_actions=1,dropout_emb=0.,
 disable_task_interaction={disable_interaction!r},enable_region_grounding={region!r},
 enable_multi_hypothesis={hypothesis!r})
print(json.dumps(sorted(ET(a).state_dict())))
"""
    output = subprocess.check_output([PYTHON, '-c', code], cwd=ROOT / worktree, text=True)
    return set(json.loads(output.strip().splitlines()[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    grounding = model_keys('03-grounding', region=True)
    grounding_off = model_keys('03-grounding', region=False)
    combined = model_keys('04-combined', region=True)
    hypothesis = model_keys('05-hypotheses', hypothesis=True)
    hypothesis_off = model_keys('05-hypotheses', hypothesis=False)
    bidir_on = model_keys('06-bidir', disable_interaction=False)
    bidir_off = model_keys('06-bidir', disable_interaction=True)

    grounding_extra = grounding - grounding_off
    hypothesis_extra = hypothesis - hypothesis_off
    checks = {
        'grounding_to_combined_exact': grounding == combined,
        'grounding_off_accepts_enabled_checkpoint_extras': (
            grounding_off < grounding and all(key.startswith('region_grounding.') for key in grounding_extra)),
        'hypothesis_off_accepts_enabled_checkpoint_extras': (
            hypothesis_off < hypothesis and all(key.startswith('hypothesis_offset_head.')
                                                for key in hypothesis_extra)),
        'interaction_toggle_does_not_change_checkpoint_keys': bidir_on == bidir_off,
        'interaction_parameters_present': any(key.startswith('task_interaction.') for key in bidir_on),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    report = {
        'checks': checks,
        'key_counts': {
            'grounding_enabled': len(grounding), 'grounding_disabled': len(grounding_off),
            'combined_enabled': len(combined), 'hypothesis_enabled': len(hypothesis),
            'hypothesis_disabled': len(hypothesis_off), 'bidirectional': len(bidir_on),
        },
        'variant_only_key_counts': {
            'grounding': len(grounding_extra), 'hypothesis': len(hypothesis_extra),
        },
        'model_source_sha256': {
            name: sha256(ROOT / name / 'multiagent/models/ET_haa.py')
            for name in ('03-grounding', '04-combined', '05-hypotheses', '06-bidir')
        },
        'interpretation': (
            'Grounding checkpoints load exactly in combined; disabled ablations only drop variant-specific '
            'extra keys. The bidirectional flag changes execution, not key layout.'
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
