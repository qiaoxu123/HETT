#!/usr/bin/env python3
"""Check an actual rollout against the exact cached stop intervention."""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from multiagent.arrival import LogisticArrivalPolicy
from scripts.arrival_analysis import stop_replay,official_item,load_rollout


def read(path):
    return load_rollout(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--online',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    policy=ap.add_mutually_exclusive_group(required=True)
    policy.add_argument('--head',type=Path)
    policy.add_argument('--distance-stop',type=float)
    ap.add_argument('--head-threshold',type=float)
    opt=ap.parse_args()
    if opt.head_threshold is not None and not opt.head:raise ValueError('Threshold override requires --head')
    policy=LogisticArrivalPolicy.load(opt.head,opt.head_threshold) if opt.head else lambda f:(f['predicted_distance_m']<=opt.distance_stop,None)
    result={}
    for split in ('val_seen','val_unseen'):
        baseline=read(opt.baseline/f'{split}.json.gz')['episodes']
        online={tuple(e['episode_id']):e for e in read(opt.online/f'{split}.json.gz')['episodes']}
        if set(online)!={tuple(e['episode_id']) for e in baseline}:
            raise RuntimeError('Episode sets differ')
        differences=[];mismatches=[]
        for e in baseline:
            replay=stop_replay(e,policy)
            actual=online[tuple(e['episode_id'])]
            a,b=np.asarray(actual['path_xy']),np.asarray(replay['path_xy'])
            if a.shape!=b.shape:
                mismatches.append(e['episode_id']);continue
            differences.append(float(np.max(np.abs(a-b))))
            if not np.allclose(a,b,atol=1e-5,rtol=0):
                mismatches.append(e['episode_id'])
        result[split]=dict(episodes=len(baseline),mismatches=mismatches,
            maximum_coordinate_difference_m=max(differences,default=0.))
        if mismatches:
            raise RuntimeError(f'Online/cached prefix mismatch: {split}, {mismatches[:5]}')
    opt.output.parent.mkdir(parents=True,exist_ok=True)
    opt.output.write_text(json.dumps(result,indent=2)+'\n')
    print('ONLINE_REPLAY_VERIFIED',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
