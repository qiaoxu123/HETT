#!/usr/bin/env python3
"""Frozen B_goal_soft_refined rollouts with lightweight causal state records."""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'multiagent'), str(ROOT/'scripts')]


def save(path, data):
    from run_joint_goal_trajectory_experiment import clean
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.gz':
        with gzip.open(str(path)+'.tmp', 'wt', compresslevel=1) as f:
            json.dump(clean(data), f, allow_nan=False, separators=(',', ':'))
        Path(str(path)+'.tmp').replace(path)
    else:
        path.write_text(json.dumps(clean(data), indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--splits', nargs='+', default=['val_seen', 'val_unseen'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--head', type=Path)
    ap.add_argument('--head-threshold',type=float,
                    help='Explicit diagnostic threshold override, requires --head')
    ap.add_argument('--semantic', action='store_true',
                    help='Cache frozen joint instruction/current RGB features')
    ap.add_argument('--distance-stop', type=float,
                    help='Observable distance baseline (mutually exclusive with --head)')
    opt = ap.parse_args()
    source, out = opt.source_run.resolve(), opt.output.resolve()
    head_path = opt.head.resolve() if opt.head else None
    if head_path and opt.distance_stop is not None:
        raise ValueError('Choose one stop policy')
    if opt.head_threshold is not None and not head_path:
        raise ValueError('Threshold override requires a trained arrival head')
    config = json.loads((ROOT/'configs/experiments/hett_arrival_aware.json').read_text())
    for key,value in {'seed':0,'batch_size':16,'max_action_len':20,
                      'max_step_m':50,'success_radius_m':20,'history_window':5}.items():
        if config[key]!=value:
            raise ValueError(f'Fixed arrival protocol changed: {key}')
    manifest = json.loads((source/'manifest.json').read_text())
    checkpoint = source/'checkpoints/epoch01.pt'
    expected = json.loads((source/'train_epoch01.json').read_text())['sha256']
    from run_joint_goal_trajectory_experiment import sha, seed, strict_init, clean
    import torch
    from parser import parse_args
    from agent import NavCMTAgent
    from env import CityNavBatch
    from torch.utils.data import DataLoader
    from joint_experiment_metrics import summarize
    from multiagent.arrival import LogisticArrivalPolicy, install_semantic_observer
    actual = sha(checkpoint)
    if actual != expected or actual != config['checkpoint_sha256']:
        raise RuntimeError('Best batch16 checkpoint SHA-256 mismatch')
    torch.set_num_threads(1)
    sys.argv = ['arrival_rollouts']
    args = parse_args()
    for k,v in manifest['args'].items():
        if hasattr(args,k):
            setattr(args,k,v)
    for k,v in manifest['config']['variants']['B_goal_soft_refined'].items():
        setattr(args,k,v)
    args.mode = 'train'
    args.resume_optimizer = False
    args.batch_size = 16
    args.seed = 0
    args.max_action_len = 20
    args.heatmap_waypoint_step_m = 50.
    args.success_dist = 20.
    args.trajectory_fast_eval = True
    args.waypoint_arrival_lock_m = 0.
    args.waypoint_switch_margin = 0.
    seed(args.seed)
    agent = NavCMTAgent(args, allow_ngpus=False, rank=0)
    compatibility = strict_init(agent, checkpoint)
    if any(r['missing_new_parameters'] or r['unexpected'] for r in compatibility.values()):
        raise RuntimeError('Arrival experiment requires exact weight compatibility')
    for model in (agent.lang_model, agent.vision_model, agent.vln_model):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    def guard(module, inputs, kwargs):
        if kwargs.get('trajectory_teacher_goal') is not None:
            raise RuntimeError('GT-conditioned model forward forbidden')
    agent.vln_model_without_ddp.register_forward_pre_hook(guard, with_kwargs=True)
    agent.arrival_policy = LogisticArrivalPolicy.load(head_path,opt.head_threshold) if head_path else None
    if opt.semantic or getattr(agent.arrival_policy, 'requires_semantic', False):
        install_semantic_observer(agent)
    if head_path and json.loads(head_path.read_text())['checkpoint_sha256']!=actual:
        raise ValueError('Arrival head was trained against another navigation checkpoint')
    if opt.distance_stop is not None:
        if not 0 < opt.distance_stop <= 20:
            raise ValueError('Distance-stop baseline must be in (0,20] meters')
        agent.arrival_policy = lambda f: (f['predicted_distance_m'] <= opt.distance_stop, None)
    out.mkdir(parents=True, exist_ok=True)
    save(out/'manifest.json', dict(source_run=str(source), checkpoint=str(checkpoint),
        checkpoint_sha256=actual, weight_compatibility=compatibility, config=config,
        args=vars(args), head=str(head_path) if head_path else None,
        distance_stop=opt.distance_stop,
        head_threshold=opt.head_threshold,
        semantic_features=opt.semantic or getattr(agent.arrival_policy, 'requires_semantic', False),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/'multiagent/agent.py',ROOT/'multiagent/arrival.py', Path(__file__)]},
        limit=opt.limit, started=time.time(), all_navigation_parameters_frozen=True))
    rows=[]
    for split in opt.splits:
        if split not in ('train_seen','val_seen','val_unseen'):
            raise ValueError('Only declared training/validation splits are allowed')
        dest=out/f'{split}.json.gz'
        if dest.exists():
            print('SKIP_EXISTING',dest,flush=True)
            continue
        seed(args.seed)
        start=time.perf_counter()
        env=CityNavBatch(split,args,batch_size=16,seed=0,rank=0,world_size=1)
        if opt.limit:
            env.data=env.data[:opt.limit]
        save(out/f'{split}_episode_ids.json',[list(e.id) for e in env.data])
        agent.env=env
        agent.experiment_step_callback=None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        load_seconds=time.perf_counter()-start
        start=time.perf_counter()
        agent.test(DataLoader(env,batch_size=1),env_name=split,feedback='student')
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        result=summarize(env,agent.get_results(),'B_goal_soft_refined',1,seconds)
        if len(result['episodes']) != len(env.data):
            raise RuntimeError('Episode coverage mismatch')
        result['summary'].update(load_seconds=load_seconds,
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
        # Annotation labels remain separate from the feature dictionary.
        rows.append(result['summary'])
        if opt.semantic or getattr(agent.arrival_policy, 'requires_semantic', False):
            import numpy as np
            from multiagent.arrival import SEMANTIC_NAMES
            states=[s for e in result['episodes'] for s in e['navigation_steps']]
            first=np.array([e['navigation_steps'][0]['semantic_features'] for e in result['episodes'][:16]])
            swapped=np.roll(first,1,axis=0)
            sensitivity=dict(diagnostic_only=True,first_batch_split=split,
                instruction_swap_max_change=float(np.abs(first[:,:768]-swapped[:,:768]).max()),
                rgb_swap_max_change=float(np.abs(first[:,768:]-swapped[:,768:]).max()),
                semantic_source='frozen BERT raw lexical embeddings and current Darknet RGB',
                extra_encoder_forwards=0)
            if min(sensitivity['instruction_swap_max_change'],sensitivity['rgb_swap_max_change'])<=1e-6:
                raise RuntimeError('Instruction/RGB descriptors must both retain observable differences')
            save(out/f'{split}_modality_sensitivity.json',sensitivity)
            cache=np.empty((len(states),len(SEMANTIC_NAMES)),dtype=np.float32)
            for j,s in enumerate(states):
                cache[j]=s.pop('semantic_features')
                s['semantic_index']=j
            cache_path=out/f'{split}_semantic.npy'
            with open(str(cache_path)+'.tmp','wb') as f:
                np.save(f,cache,allow_pickle=False)
            Path(str(cache_path)+'.tmp').replace(cache_path)
            result['semantic_cache']=dict(file=cache_path.name,sha256=sha(cache_path),
                feature_names=list(SEMANTIC_NAMES),states=len(states),dtype='float32')
            del cache,states
        save(dest,result)
        save(out/f'{split}_metrics.json',result['summary'])
        print('ARRIVAL_COLLECTION_COMPLETE',split,json.dumps(clean(result['summary'])),flush=True)
    save(out/'COMPLETE.json',dict(finished=time.time(),splits=opt.splits))


if __name__=='__main__':
    main()
