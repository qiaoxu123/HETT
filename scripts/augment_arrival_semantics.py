#!/usr/bin/env python3
"""Extract frozen lexical/RGB features at saved causal poses, without replanning."""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'multiagent'),str(ROOT/'scripts')]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-run',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--splits',nargs='+',default=['train_seen','val_seen','val_unseen'])
    ap.add_argument('--compare',type=Path)
    opt=ap.parse_args()
    source,baseline,out=opt.source_run.resolve(),opt.baseline.resolve(),opt.output.resolve()
    compare=opt.compare.resolve() if opt.compare else None
    from collect_arrival_rollouts import save
    from run_joint_goal_trajectory_experiment import sha,seed,strict_init
    import numpy as np
    import torch
    from parser import parse_args
    from agent import NavCMTAgent
    from env import CityNavBatch
    from multiagent.arrival import semantic_descriptor,lexical_instruction_features,SEMANTIC_NAMES
    from scripts.arrival_analysis import load_rollout
    from multiagent.observation import cropclient
    from multiagent.space import Pose4D
    config=json.loads((ROOT/'configs/experiments/hett_arrival_aware.json').read_text())
    checkpoint=source/'checkpoints/epoch01.pt'
    if sha(checkpoint)!=config['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
    manifest=json.loads((baseline/'manifest.json').read_text())
    if manifest['checkpoint_sha256']!=config['checkpoint_sha256']:raise ValueError('Baseline mismatch')
    sys.argv=['augment_arrival_semantics'];args=parse_args()
    for k,v in manifest['args'].items():
        if hasattr(args,k):setattr(args,k,v)
    torch.set_num_threads(1);seed(0)
    agent=NavCMTAgent(args,allow_ngpus=False,rank=0)
    compatibility=strict_init(agent,checkpoint)
    if any(r['missing_new_parameters'] or r['unexpected'] for r in compatibility.values()):
        raise ValueError('Weight incompatibility')
    for model in (agent.lang_model,agent.vision_model,agent.vln_model):
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
    out.mkdir(parents=True,exist_ok=True)
    save(out/'manifest.json',dict(manifest,semantic_features=True,config=config,
        semantic_extraction='current RGB at cached pose, current instruction only',
        semantic_code_sha256=sha(Path(__file__)),navigation_recomputed=False))
    resources=[]
    for split in opt.splits:
        if split not in ('train_seen','val_seen','val_unseen'):raise ValueError(split)
        dest=out/f'{split}.json.gz'
        if dest.exists():raise ValueError(f'Refuse to overwrite {dest}')
        start=time.perf_counter();seed(0)
        env=CityNavBatch(split,args,batch_size=16,seed=0,rank=0,world_size=1)
        # No target positions, teacher paths or future poses enter encoders.
        observable={tuple(e.id):(e.target_description,e.map_name) for e in env.data}
        with gzip.open(baseline/f'{split}.json.gz','rt') as f:data=json.load(f)
        episodes=data['episodes'];states=sum(len(e['navigation_steps']) for e in episodes)
        cache_path=out/f'{split}_semantic.npy'
        cache=np.lib.format.open_memmap(str(cache_path)+'.tmp',mode='w+',dtype=np.float32,
                                       shape=(states,len(SEMANTIC_NAMES)))
        index=0;torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for offset in range(0,len(episodes),16):
                batch=episodes[offset:offset+16]
                encoding=agent.tokenizer([observable[tuple(e['episode_id'])][0] for e in batch],
                                         padding=True,return_tensors='pt')
                language=lexical_instruction_features(agent.lang_model,
                    encoding['input_ids'].cuda(),encoding['attention_mask'].cuda())
                # Layout is episode-major, matching the saved state indices.
                starts=[]
                for e in batch:
                    starts.append(index);index+=len(e['navigation_steps'])
                for t in range(max(len(e['navigation_steps']) for e in batch)):
                    active=[i for i,e in enumerate(batch) if t<len(e['navigation_steps'])]
                    nav=[batch[i]['navigation_steps'][t] for i in active]
                    images=np.stack([cropclient.crop_image(observable[tuple(batch[i]['episode_id'])][1],
                        Pose4D(*s['pose']),(224,224),'rgb') for i,s in zip(active,nav)])
                    images=np.ascontiguousarray(images[:,:,:,::-1].transpose(0,3,1,2),dtype=np.float32)
                    images=(images-agent.rgb_mean)/agent.rgb_std
                    rgb=agent.vision_model(torch.from_numpy(images).cuda())
                    values=semantic_descriptor(language[active],rgb).cpu().numpy()
                    for i,s,v in zip(active,nav,values):
                        j=starts[i]+t;cache[j]=v;s['semantic_index']=j
                if offset%160==0:
                    print('SEMANTIC_REPLAY_PROGRESS',split,offset+len(batch),len(episodes),
                          round(time.perf_counter()-start,2),flush=True)
        torch.cuda.synchronize();cache.flush();del cache
        Path(str(cache_path)+'.tmp').replace(cache_path)
        data['semantic_cache']=dict(file=cache_path.name,sha256=sha(cache_path),states=states,
                                    feature_names=list(SEMANTIC_NAMES),dtype='float32')
        save(dest,data)
        row=dict(split=split,episodes=len(episodes),states=states,seconds=time.perf_counter()-start,
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),all_backbones_frozen=True,
            baseline_sha256=sha(baseline/f'{split}.json.gz'),navigation_recomputed=False)
        if compare:
            reference={tuple(e['episode_id']):e for e in load_rollout(compare/f'{split}.json.gz')['episodes']}
            augmented=load_rollout(dest);difference=0.
            for e in augmented['episodes']:
                a=np.array([s['semantic_features'] for s in e['navigation_steps']])
                b=np.array([s['semantic_features'] for s in reference[tuple(e['episode_id'])]['navigation_steps']])
                if a.shape!=b.shape or not np.allclose(a,b,atol=1e-5,rtol=0):
                    raise RuntimeError('Cached-pose features disagree with live current observation')
                difference=max(difference,float(np.abs(a-b).max()))
            row['maximum_live_feature_difference']=difference
        resources.append(row);save(out/f'{split}_semantic_replay.json',row)
        save(out/f'{split}_episode_ids.json',[e['episode_id'] for e in episodes])
        print('SEMANTIC_REPLAY_COMPLETE',json.dumps(row),flush=True)
    save(out/'COMPLETE.json',dict(resources=resources,finished=time.time()))


if __name__=='__main__':main()
