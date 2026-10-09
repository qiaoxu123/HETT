#!/usr/bin/env python3
"""Teacher-forced, no-update audit of candidate ranking coverage."""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'multiagent'), str(ROOT), str(ROOT / 'scripts')]
from parser import parse_args
from env import CityNavBatch
from run_joint_goal_trajectory_experiment import new_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--episodes', type=int, default=128)
    opt = ap.parse_args()
    cfg = json.loads(opt.config.read_text())
    sys.argv = ['training_supervision_audit']
    args = parse_args()
    base = json.loads(Path(cfg['base_arguments']).read_text())
    for k, v in base.items():
        if hasattr(args, k): setattr(args, k, v)
    args.mode = 'train'; args.batch_size = 1; args.seed = cfg['seed']
    args.heatmap_trajectory_enabled = True; args.trajectory_goal_k = 5
    args.trajectory_relation_selector = False; args.max_action_len = 20
    args.trajectory_train_diagnostics = True
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    agent, init_report = new_agent(args, cfg)
    agent.load(str(opt.checkpoint.resolve()))
    env = CityNavBatch('train_seen', args, batch_size=1, seed=args.seed, rank=0, world_size=1)
    rng = np.random.RandomState(args.seed + 71)
    indices = rng.choice(len(env.data), min(opt.episodes, len(env.data)), replace=False)
    env.data = [env.data[int(i)] for i in indices]
    env.ix = 0
    agent.env = env; agent.env_name = 'train_seen'; agent.feedback = 'teacher'
    agent.lang_model.train(); agent.vln_model.train(); agent.vision_model.train()
    agent.logs = defaultdict(list)
    torch.cuda.reset_peak_memory_stats()
    for _ in env:
        agent.feedback = 'teacher'
        agent.loss = 0
        with torch.no_grad():
            agent.rollout(train_ml=args.teacher_weight)
        agent.loss = 0
    vals = {k: float(sum(float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
                         for x in v))
            for k, v in agent.logs.items() if k.startswith('trajectory_') and v}
    active = vals.get('trajectory_ranking_active_count', 0.)
    rank_valid = vals.get('trajectory_ranking_valid_count', 0.)
    count = vals.get('trajectory_eval_count', 0.)
    report = {
        'checkpoint': str(opt.checkpoint.resolve()),
        'episodes': len(env.data), 'candidate_k': args.trajectory_goal_k,
        'seed': args.seed, 'teacher_yaw_randomization': '50% chance at every teacher observation',
        'teacher_supervision_active_steps': active,
        'ranking_valid_steps': rank_valid,
        'ranking_valid_fraction': rank_valid / max(1., active),
        'nearest_candidate_gt_error_mean_m': vals.get('trajectory_candidate_goal_error_sum_m', 0.) / max(1., active),
        'proposal_eval_steps': count,
        'minADE_mean_m': vals.get('trajectory_minade_sum_m', 0.) / max(1., count),
        'minFDE_mean_m': vals.get('trajectory_minfde_sum_m', 0.) / max(1., count),
        'prior_goal_hit_fraction': vals.get('trajectory_prior_goal_hits', 0.) / max(1., count),
        'joint_goal_hit_fraction': vals.get('trajectory_joint_goal_hits', 0.) / max(1., count),
        'oracle_candidate_hit_fraction': vals.get('trajectory_oracle_goal_hits', 0.) / max(1., count),
        'stop_positive_fraction': vals.get('trajectory_stop_positive_count', 0.) / max(1., vals.get('trajectory_stop_supervised_count', 0.)),
        'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved(),
        'optimizer_steps': 0, 'optimizer_state_changed': False,
        'initial_checkpoint_load': init_report,
    }
    opt.output.parent.mkdir(parents=True, exist_ok=True)
    opt.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
