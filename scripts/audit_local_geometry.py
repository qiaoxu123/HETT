"""Read-only data checks and a ground-truth perturbation audit for the probe."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_local_geometry import (ROOT, OLD, SIDE, Cropper, Localizer, make_args,
                                 select_tasks, sha, write_json, SEEDS)
import numpy as np
import torch


def data_audit(out):
    manifests=[json.loads((out/f'{s}_manifest.json').read_text()) for s in ['train_seen','val_unseen']]
    train_maps=set(manifests[0]['maps']); val_maps=set(manifests[1]['maps'])
    assert not train_maps & val_maps
    report={'map_overlap':sorted(train_maps&val_maps),'checks':[], 'files':{}}
    for split, manifest in zip(['train_seen','val_unseen'],manifests):
        features=dict(np.load(out/f'{split}_features.npz'))
        labels=np.load(out/f'{split}_labels.npy')
        assert set(features)=={'rgb','geo','text'}
        assert all(np.isfinite(x).all() and len(x)==len(labels) for x in features.values())
        assert len(labels)==len(manifest['samples'])
        assert all(x['step']<=18 for x in manifest['samples'])
        report['checks'].append(dict(split=split,states=len(labels),finite_features=True,
                                     feature_shapes={k:list(v.shape) for k,v in features.items()}))
        for suffix in ['features.npz','labels.npy','manifest.json','trajectories.pt']:
            path=out/f'{split}_{suffix}'
            report['files'][path.name]=sha(path)
    # Check binary record lengths; raw semantic fields are not consumed by probe.
    plyroot=Path('/home/tenant2/Workspace/DATA/SensatUrban_Dataset/ply')
    bad=[]
    for name in sorted(train_maps|val_maps):
        path=next(plyroot.glob(f'*/{name}.ply'))
        with path.open('rb') as f:
            header=[]
            while True:
                line=f.readline().decode().strip();header.append(line)
                if line=='end_header':break
            offset=f.tell()
        n=int(next(x.split()[-1] for x in header if x.startswith('element vertex')))
        size=16 if 'property uint8 class' in header else 15
        if path.stat().st_size != offset+n*size:bad.append(name)
    assert not bad
    report['ply_length_failures']=bad
    write_json(out/'data_audit.json',report)
    print('DATA AUDIT PASSED',report['checks'],flush=True)


@torch.no_grad()
def policy_audit(out, checkpoint):
    import multiagent.env as em
    from multiagent.agent import NavCMTAgent
    from multiagent.space import Point2D
    torch.set_num_threads(2)
    args=make_args(checkpoint)
    original_loader=em.load_mturk_trajectories
    tasks=select_tasks(original_loader('val_unseen','all',args.altitude),2)
    em.load_mturk_trajectories=lambda *a,**k:tasks
    env=em.CityNavBatch('val_unseen',args,batch_size=2,seed=0)
    em.load_mturk_trajectories=original_loader
    model=NavCMTAgent(args,rank=-1,allow_ngpus=False);model.load(str(checkpoint))
    model.env=env;model.feedback='student';model.env_name='test_audit'
    for m in [model.lang_model,model.vision_model,model.vln_model]:m.eval()
    next(iter(env)); model.loss=0
    original=model.rollout()
    get_obs=env._get_obs
    def poisoned(*a,**kw):
        obs=get_obs(*a,**kw)
        for ob in obs:
            ob['goal']=Point2D(98765.,-98765.)
            ob['normalized_goal']=(.99,.01)
            ob['direction']=2.345
            ob['progress']=.999
            ob['grid_goal']=24
        return obs
    env._get_obs=poisoned
    next(iter(env)); model.loss=0
    perturbed=model.rollout()
    differences=[]
    for a,b in zip(original,perturbed):
        pa,pb=np.asarray(a['trajectory']),np.asarray(b['trajectory'])
        assert pa.shape==pb.shape
        diff=float(np.abs(pa-pb).max());differences.append(diff)
        assert diff<1e-6, 'GT perturbation changed student trajectory'
    write_json(out/'gt_perturbation_audit.json',dict(episodes=len(differences),
               max_trajectory_difference=differences,
               perturbed_fields=['goal','normalized_goal','direction','progress','grid_goal'],
               limitation='Checks student control dependence on these returned GT fields; not a proof of all pipeline correctness.'))
    print('POLICY GT PERTURBATION PASSED',differences,flush=True)


@torch.no_grad()
def geometry_audit(out):
    from multiagent.mapdata import GROUND_LEVEL
    torch.set_num_threads(2)
    meta=json.loads((out/'evaluation_manifest.json').read_text())
    labels=np.load(out/'evaluation_labels.npy')
    inside=[]
    for item,(dx,negdy) in zip(meta['samples'],labels):
        _,_,z,yaw=item['pose']; dy=-negdy; h=z-GROUND_LEVEL[item['map']]
        inside.append(abs(dx*np.cos(yaw)+dy*np.sin(yaw))<=h and
                      abs(-dx*np.sin(yaw)+dy*np.cos(yaw))<=h)
    train=dict(np.load(out/'train_seen_features.npz'))
    norm=dict(np.load(out/'normalization.npz'))
    features={k:torch.tensor((v[:16]-norm[k+'_mean'])/norm[k+'_std']) for k,v in train.items()}
    a,b=np.meshgrid((np.arange(7)+.5)/7-.5,(np.arange(7)+.5)/7-.5)
    coords=torch.tensor(np.stack([a.ravel(),b.ravel()],1),dtype=torch.float32)
    predictions=dict(np.load(out/'predictions.npz'))
    checks=[]
    for seed in SEEDS:
        torch.manual_seed(seed);model=Localizer();initial=model.geometry.weight.detach().clone()
        model.load_state_dict(torch.load(out/f'rgb_geo_text_seed{seed}.pt',weights_only=True,map_location='cpu'))
        model.eval()
        normal=model(features['rgb'],features['geo'],features['text'],coords)
        zero=model(features['rgb'],torch.zeros_like(features['geo']),features['text'],coords)
        change=float((normal-zero).abs().max());weight=float((model.geometry.weight-initial).abs().max())
        assert change>0 and weight>0
        p=predictions[f'rgb_geo_text/normal/seed{seed}/prediction']
        q=predictions[f'rgb_geo_text/shuffled_text/seed{seed}/prediction']
        checks.append(dict(seed=seed,max_geometry_weight_change=weight,
                           max_logits_change_zero_geometry=change,
                           positions_changed_by_shuffled_text=int((np.abs(p-q)>1e-6).any(1).sum())))
    write_json(out/'geometry_usage_audit.json',dict(checks=checks,
               full_camera_ground_footprint_target_coverage_pct=float(np.mean(inside)*100),
               stage2_started_pct=float(np.mean([s['stage2_started'] for s in meta['samples']])*100)))
    print('GEOMETRY AUDIT PASSED',checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--policy',action='store_true')
    p.add_argument('--geometry',action='store_true')
    a=p.parse_args()
    if a.policy:policy_audit(a.out,a.checkpoint)
    elif a.geometry:geometry_audit(a.out)
    else:data_audit(a.out)
