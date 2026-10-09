#!/usr/bin/env python3
"""Oracle diagnostics, train-only logistic fit and seen-only calibration."""
import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from multiagent.arrival import (FEATURE_NAMES, JOINT_FEATURE_NAMES, SEMANTIC_NAMES,
                               LogisticArrivalPolicy, arrival_inputs)
from multiagent.arrival import MLPArrivalPolicy
from scripts.arrival_analysis import (metrics, official_item, oracle_stop, oracle_goal,
    episode_diagnostics, stop_replay, stop_metrics, load_rollout)


def load(path):
    return load_rollout(path)


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def scalar(value):
        if isinstance(value,np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=scalar)+'\n')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def csv_save(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict,tuple)) else v
                         for k,v in r.items()} for r in rows)


def feature_dataset(episodes, feature_names=FEATURE_NAMES):
    if feature_names not in (FEATURE_NAMES,JOINT_FEATURE_NAMES):
        raise ValueError('Only declared observable feature schemas may train arrival')
    total=sum(len(e['navigation_steps']) for e in episodes)
    x=np.empty((total,len(feature_names)),dtype=np.float64)
    y=[]; weight=[];index=0
    for e in episodes:
        states=e['navigation_steps']
        for s in states:
            if set(s['features'])!=set(FEATURE_NAMES):
                raise ValueError('Observable schema mismatch')
            x[index,:len(FEATURE_NAMES)]=[s['features'][n] for n in FEATURE_NAMES]
            if feature_names == JOINT_FEATURE_NAMES:
                if len(s.get('semantic_features', [])) != len(SEMANTIC_NAMES):
                    raise ValueError('Missing current instruction/RGB features')
                x[index,len(FEATURE_NAMES):]=s['semantic_features']
            index+=1
            # Label only the current decision pose, never the next pose.
            y.append(np.linalg.norm(np.asarray(s['pose'][:2])-e['goal_xy'])<=20.)
            weight.append(1./len(states))
    weights=np.asarray(weight)
    # Equal episode weighting, normalized to mean one for fixed regularization.
    return x,np.asarray(y,dtype=int),weights/weights.mean()


class CalibrationCache:
    """Fast threshold evaluation, all prefix scores from the official evaluator."""
    def __init__(self,episodes,probabilities):
        self.rows=[]
        offset=0
        for e in episodes:
            nav=e['navigation_steps']
            probs=probabilities[offset:offset+len(nav)];offset+=len(nav)
            label=np.array([np.linalg.norm(np.asarray(s['pose'][:2])-e['goal_xy'])<=20. for s in nav])
            prefix={}
            for s in nav:
                i=s['path_index']
                if i not in prefix:
                    prefix[i]=official_item(e,e['path_xy'][:i+1])
            self.rows.append((e,probs,label,prefix,official_item(e)))

    def evaluate(self,threshold):
        sr=[];spl=[];ne=[];osr=[];tp=fp=fn=tn=eligible=lost=0
        for e,probs,labels,prefix,base in self.rows:
            hits=np.flatnonzero(probs>=threshold)
            stop=int(hits[0]) if len(hits) else None
            visited=labels[:stop+1] if stop is not None else labels
            positive=int(stop is not None and labels[stop]);negative=int(stop is not None and not labels[stop])
            tp+=positive;fp+=negative;fn+=int(visited.sum())-positive
            tn+=len(visited)-int(visited.sum())-negative;eligible+=int(labels.any())
            lost+=int(negative and base['success'])
            scores=prefix[e['navigation_steps'][stop]['path_index']] if stop is not None else base
            sr.append(scores['success']);spl.append(scores['spl']);ne.append(scores['ne']);osr.append(scores['oracle_success'])
        n=len(sr)
        return dict(episodes=n,sr=100*float(np.mean(sr)),spl=100*float(np.mean(spl)),
            oracle_sr=100*float(np.mean(osr)),ne=float(np.mean(ne)),
            stop_tp=tp,stop_fp=fp,stop_fn=fn,stop_tn=tn,
            stop_precision=tp/(tp+fp) if tp+fp else None,
            stop_recall=tp/(tp+fn) if tp+fn else None,
            stop_episode_recall=tp/eligible if eligible else None,
            false_stop_episode_rate=fp/n,false_positive_state_rate=fp/(fp+tn) if fp+tn else None,
            false_stop_replaced_success=lost,policy_stops=tp+fp)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rollouts',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--oracle-only',action='store_true')
    ap.add_argument('--semantic',action='store_true',
                    help='Train a joint instruction/RGB and belief logistic head')
    opt=ap.parse_args()
    out=opt.output;out.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/experiments/hett_arrival_aware.json').read_text())
    start=time.perf_counter()
    split_data={s:load(opt.rollouts/f'{s}.json.gz') for s in ('val_seen','val_unseen')}
    rows=[];diagnostics=[];feature_diagnostics=[]
    for split,data in split_data.items():
        episodes=data['episodes']
        variants={'A_original':episodes,'B_oracle_stop':[oracle_stop(e) for e in episodes],
            'C_oracle_goal':[oracle_goal(e) for e in episodes],
            'D_oracle_goal_stop':[oracle_goal(e,stop=True) for e in episodes]}
        for name,ep in variants.items():
            row=dict(split=split,variant=name,**metrics(ep))
            csv_save(out/f'{split}_{name}_episodes.csv.gz',[
                dict(episode_id=e['episode_id'],termination=e['termination'],
                     path_xy=json.dumps(e['path_xy']),**official_item(e)) for e in ep])
            if name=='A_original':
                row.update({k:data['summary'][k] for k in ('goal_switch_count','goal_switch_rate',
                    'horizon_timeouts','waypoint_stagnations','seconds',
                    'peak_vram_allocated_bytes','peak_vram_reserved_bytes')})
                positives=sum(np.linalg.norm(np.asarray(s['pose'][:2])-e['goal_xy'])<=20.
                              for e in episodes for s in e['navigation_steps'])
                states=sum(len(e['navigation_steps']) for e in episodes)
                row.update(stop_tp=0,stop_fp=0,stop_fn=int(positives),stop_tn=states-int(positives),
                           stop_precision=None,stop_recall=0.,false_stop_episode_rate=0.)
            if name in ('B_oracle_stop','D_oracle_goal_stop'):
                stopped=sum(e['termination']=='oracle_stop' for e in ep)
                row.update(stop_tp=stopped,stop_fp=0,stop_precision=1. if stopped else None,
                           false_stop_episode_rate=0.,policy_stops=stopped)
            rows.append(row)
        for e in episodes:
            diagnostics.append(dict(split=split,**episode_diagnostics(e)))
        distances=np.array([s['features']['predicted_distance_m'] for e in episodes for s in e['navigation_steps']])
        labels=np.array([np.linalg.norm(np.asarray(s['pose'][:2])-e['goal_xy'])<=20.
                         for e in episodes for s in e['navigation_steps']])
        for cutoff in (1.,5.,10.,20.):
            mask=distances<=cutoff
            feature_diagnostics.append(dict(split=split,predicted_distance_cutoff_m=cutoff,
                states=int(mask.sum()),true_arrival_fraction=float(labels[mask].mean()) if mask.any() else None))
        print('ORACLE_COMPLETE',split,json.dumps(rows[-4:]),flush=True)
    csv_save(out/'oracle_metrics.csv',rows)
    csv_save(out/'observable_distance_diagnostics.csv',feature_diagnostics)
    csv_save(out/'episode_diagnostics.csv.gz',diagnostics)
    groups={}
    for split in split_data:
        ds=[r for r in diagnostics if r['split']==split]
        groups[split]={k:sum(r['failure_group']==k for r in ds) for k in ('success','never_entered','entered_then_left')}
        groups[split].update({k:sum(r[k] for r in ds) for k in ('exit_goal_switches',
            'exit_wrong_predicted_goals','exit_large_actions','exit_full_50m_actions',
            'skipped_success_circle_crossings','reentries',
            'never_entered_no_correct_goal','never_entered_despite_correct_goal')})
    save(out/'failure_groups.json',groups)
    if opt.oracle_only:
        save(out/'COMPLETE.json',dict(seconds=time.perf_counter()-start,oracle_only=True))
        return
    train_path=opt.rollouts/'train_seen.json.gz'
    train=load(train_path)
    if train['summary']['split']!='train_seen':
        raise ValueError('Only train_seen may train the arrival head')
    feature_names=JOINT_FEATURE_NAMES if opt.semantic else FEATURE_NAMES
    x,y,w=feature_dataset(train['episodes'], feature_names)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler=StandardScaler().fit(x,sample_weight=w)
    fit_start=time.perf_counter()
    if opt.semantic:
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        hparams=config['joint_arrival_head']
        torch.cuda.reset_peak_memory_stats()
        tx=torch.from_numpy(scaler.transform(x).astype(np.float32)).cuda()
        ty=torch.from_numpy(y.astype(np.float32)).cuda()
        tw=torch.from_numpy(w.astype(np.float32)).cuda()
        network=torch.nn.Sequential(torch.nn.Linear(len(feature_names),16),torch.nn.ReLU(),
                                    torch.nn.Linear(16,1)).cuda()
        optimizer=torch.optim.AdamW(network.parameters(),lr=hparams['learning_rate'],
                                    weight_decay=hparams['weight_decay'])
        losses=[]
        for epoch in range(hparams['epochs']):
            order=torch.randperm(len(ty),device='cuda');total=0.
            for begin in range(0,len(ty),hparams['batch_size']):
                indices=order[begin:begin+hparams['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                logits=network(tx[indices]).squeeze(-1)
                loss=(F.binary_cross_entropy_with_logits(logits,ty[indices],reduction='none')*tw[indices]).mean()
                loss.backward();optimizer.step();total+=float(loss.detach())*len(indices)
            losses.append(total/len(ty))
            print('JOINT_MLP_TRAIN',epoch+1,losses[-1],flush=True)
        torch.cuda.synchronize()
        model_payload=dict(head_type='mlp16',w1=network[0].weight.detach().cpu().tolist(),
            b1=network[0].bias.detach().cpu().tolist(),w2=network[2].weight.detach().cpu()[0].tolist(),
            b2=float(network[2].bias.detach().cpu()[0]),training_loss=losses,hyperparameters=hparams,
            peak_training_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_training_vram_reserved_bytes=torch.cuda.max_memory_reserved())
        del tx,ty,tw,network,optimizer
        policy_class=MLPArrivalPolicy
    else:
        model=LogisticRegression(C=config['logistic_C'],max_iter=500,random_state=config['seed'])
        model.fit(scaler.transform(x),y,sample_weight=w)
        model_payload=dict(head_type='logistic',coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]))
        policy_class=LogisticArrivalPolicy
    fit_seconds=time.perf_counter()-fit_start
    payload=dict(feature_names=list(feature_names),mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),threshold=1.01,**model_payload,
        train_split='train_seen',calibration_split='val_seen',seed=0,
        train_episodes=len(train['episodes']),train_states=len(y),positive_states=int(y.sum()),
        training_data_sha256=sha(train_path),fit_seconds=fit_seconds,
        checkpoint_sha256=json.loads((opt.rollouts/'manifest.json').read_text())['checkpoint_sha256'])
    payload['modalities']=['current_instruction','current_rgb','belief_pose_history'] if opt.semantic else ['belief_pose_history']
    payload['frozen_semantic_source']='frozen BERT raw word embeddings768; frozen current Darknet spatial pooled RGB512' if opt.semantic else None
    if opt.semantic:
        payload['training_semantic_sha256']=train['semantic_cache']['sha256']
    # Free training records before processing large validation prefix tables.
    del train,x,y,w
    head=policy_class(payload)
    seen=split_data['val_seen']['episodes']
    probs=np.array([head.probability(arrival_inputs(s,head)) for e in seen for s in e['navigation_steps']])
    seen_cache=CalibrationCache(seen,probs)
    thresholds=sorted(set([.5,.7,.8,.9,.95,.97,.99,.995,.999,1.01]+np.quantile(probs,[.5,.6,.7,.8,.9,.95,.97,.98,.99,.995,.999]).tolist()))
    baseline=metrics(seen)
    calibration=[]
    for threshold in thresholds:
        row=dict(threshold=float(threshold),**seen_cache.evaluate(threshold))
        row['admissible']=bool(row['policy_stops'] and row['stop_precision']>=config['minimum_stop_precision']
            and row['false_stop_episode_rate']<=config['maximum_false_stop_episode_rate']
            and row['sr']>=baseline['sr'] and row['spl']>=baseline['spl'])
        calibration.append(row)
    accepted=[r for r in calibration if r['admissible']]
    selected=max(accepted,key=lambda r:(r['sr'],r['spl'],-r['stop_fp'])) if accepted else None
    payload['threshold']=selected['threshold'] if selected else 1.01
    payload['calibration_status']='accepted' if selected else 'no_safe_seen_threshold_disable_stop'
    payload['threshold_candidates']=thresholds
    payload['false_positive_constraints']=dict(minimum_precision=config['minimum_stop_precision'],
        maximum_false_stop_episode_rate=config['maximum_false_stop_episode_rate'],
        require_seen_sr_non_decreasing=True,require_seen_spl_non_decreasing=True)
    save(out/'arrival_head.json',payload)
    csv_save(out/'seen_threshold_calibration.csv',calibration)
    head=policy_class(payload)
    policy_rows=[];classifier_rows=[]
    # Unseen is opened for reporting only AFTER the seen threshold is frozen.
    for split,data in split_data.items():
        episodes=data['episodes']
        from sklearn.metrics import roc_auc_score,average_precision_score
        state_probs=np.array([head.probability(arrival_inputs(s,head)) for e in episodes for s in e['navigation_steps']])
        state_labels=np.array([np.linalg.norm(np.asarray(s['pose'][:2])-e['goal_xy'])<=20.
                               for e in episodes for s in e['navigation_steps']])
        classifier_rows.append(dict(split=split,states=len(state_labels),
            positive_rate=float(state_labels.mean()),roc_auc=float(roc_auc_score(state_labels,state_probs)),
            average_precision=float(average_precision_score(state_labels,state_probs)),
            positive_score_median=float(np.median(state_probs[state_labels])),
            negative_score_median=float(np.median(state_probs[~state_labels])),
            maximum_score=float(state_probs.max())))
        policies=[('distance_stop_20m',lambda f:(f['predicted_distance_m']<=20.,None)),
                  ('arrival_joint_mlp16' if opt.semantic else 'arrival_logistic',head)]
        if opt.semantic:
            policies.append(('arrival_joint_mlp16_fixed_0.5_diagnostic',
                MLPArrivalPolicy(payload,threshold=config['joint_arrival_head']['diagnostic_threshold'])))
        for name,policy in policies:
            ep=[stop_replay(e,policy) for e in episodes]
            row=dict(split=split,variant=name,**metrics(ep),**stop_metrics(ep))
            policy_rows.append(row)
            csv_save(out/f'{split}_{name}_episodes.csv.gz',[
                dict(episode_id=e['episode_id'],termination=e['termination'],
                     **official_item(e),**e['stop_diagnostics']) for e in ep])
        print('ARRIVAL_HEAD_COMPLETE',split,json.dumps(policy_rows[-2:]),flush=True)
    csv_save(out/'ablation_metrics.csv',rows+policy_rows)
    csv_save(out/'classifier_diagnostics.csv',classifier_rows)
    save(out/'COMPLETE.json',dict(seconds=time.perf_counter()-start,head_fit_seconds=fit_seconds,
        head_sha256=sha(out/'arrival_head.json'),calibration_status=payload['calibration_status'],
        independent_unseen_threshold=payload['threshold']))


if __name__=='__main__':
    main()
