#!/usr/bin/env python3
"""Train statistics, val_seen selection/calibration, sealed unseen reporting."""
from __future__ import annotations
import json,sys,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.config import load_config,load_landmarks
from sensaturban_fpv.spatial_graph import build_block_graph
from sensaturban_fpv.spatial_graph.gate_f_programs import programs,score_candidates,family
from sensaturban_fpv.spatial_graph.gate_f_calibration import fit_calibration
from sensaturban_fpv.spatial_graph.node_types import NodeKind
from sensaturban_fpv.entity_geometry import group_of
OUT=ROOT/'artifacts/spatial_canonicalization_gate_f'

def metric(scores,target):
    s=np.asarray(scores,dtype=float);positive=s[target];negative=np.delete(s,target)
    if not len(negative):return None
    diff=positive-negative
    auc=float(np.mean((diff>0)+.5*(diff==0)))
    greater=int(np.sum(s>positive));equal=int(np.sum(s==positive))
    ranks=np.arange(greater+1,greater+equal+1)
    return {'auc':auc,'top1':float(np.mean(ranks<=1)),'top4':float(np.mean(ranks<=4)),
            'mrr':float(np.mean(1/ranks)),'margin':float(np.mean(diff))}

def aggregate(items):
    if not items:return None
    return {k:float(np.mean([m[k] for m in items])) for k in ('auc','top1','top4','mrr','margin')} | {'n':len(items)}

def key_of(row,context=True):
    kind=(row['anchor_type'] or 'unknown') if context else 'all'
    return (row['relation_phrase'].lower(),kind,row['clause_source'])

def norm_scores(x):
    x=np.asarray(x,dtype=float)
    sd=np.std(x)
    return (x-np.mean(x))/(sd if sd>1e-8 else 1.)

def combine(scores,probabilities,reliability=1.):
    names=[p for p,w in probabilities.items() if w>0 and (p in scores or p=='UNKNOWN')]
    if not names or not scores:return None
    weights=np.asarray([probabilities[n] for n in names],dtype=float);weights/=weights.sum()
    width=len(next(iter(scores.values())))
    stack=np.stack([np.zeros(width) if n=='UNKNOWN' else norm_scores(scores[n]) for n in names])
    z=np.log(weights)[:,None]+stack
    m=z.max(axis=0);out=m+np.log(np.exp(z-m).sum(axis=0))
    return reliability*out

def main():
    rows=[json.loads(x) for x in (OUT/'extended_geometry_rows.jsonl').open()]
    cfg=load_config();objects=load_landmarks(cfg);graphs={};road_cache={}
    evaluated=[];coverage=Counter()
    for idx,row in enumerate(rows):
        graph=graphs.get(row['map_id'])
        if graph is None:
            graph=build_block_graph(objects[row['map_id']],row['map_id']);graphs[row['map_id']]=graph
            print('graph',row['map_id'],flush=True)
        ev=row['evaluation'];candidate_ids=[int(x) for x in ev['candidate_ids']]
        target_type=group_of(graph.by_id[int(ev['target_id'])].entity_type)
        ids=[i for i in candidate_ids if i in graph.by_id and group_of(graph.by_id[i].entity_type)==target_type]
        if int(ev['target_id']) not in ids or len(ids)<2:continue
        candidates=[graph.by_id[i] for i in ids];target=ids.index(int(ev['target_id']))
        anchors=[graph.by_id[x['id']] for x in row['anchor_hypotheses'] if x['id'] in graph.by_id]
        seconds=[graph.by_id[x['id']] for x in row['second_anchor_hypotheses'] if x['id'] in graph.by_id]
        kind='road_region' if anchors and all(a.kind is NodeKind.ROAD_REGION for a in anchors) else row['anchor_type'] or 'unknown'
        progs=programs(row['relation_phrase'],kind,bool(seconds))
        scores={}
        for program in progs:
            s=score_candidates(program,row,candidates,anchors,seconds,graph,road_cache)
            if s is not None:scores[program]=s
        if not scores:continue
        old_program='OLD_NEAR' if row['relation_phrase'].lower() in ('near','next to','beside','close to','adjacent to') and kind!='road_region' else 'OLD_BETWEEN' if row['relation_phrase'].lower()=='between' and seconds else None
        old_score=score_candidates(old_program,row,candidates,anchors,seconds,graph,road_cache) if old_program else None
        measured={p:metric(s,target) for p,s in scores.items()}
        measured={p:m for p,m in measured.items() if m}
        if not measured:continue
        evaluated.append({'id':row['sample_id'],'episode_id':row['episode_id'],'split':row['split'],
                          'phrase':row['relation_phrase'].lower(),'kind':kind,'source':row['clause_source'],
                          'family':family(row['relation_phrase'],kind),'target':target,
                          'scores':{p:s.tolist() for p,s in scores.items()},'metrics':measured,
                          'protocol':ev.get('candidate_protocol','legacy_hard_negative'),
                          'old_scores':old_score.tolist() if old_score is not None else None})
        coverage['evaluated']+=1
        if idx%500==0:print('rows',idx,len(evaluated),flush=True)
    # Calibrate per phrase + anchor type and a phrase-only comparator. Unseen never enters selection.
    calibrations={};records=[]
    for context in (False,True):
        groups=defaultdict(list)
        for e in evaluated:groups[(e['phrase'],e['kind'] if context else 'all',e['source'])].append(e)
        for key,items in groups.items():
            train=[e for e in items if e['split']=='train_seen'];val=[e for e in items if e['split']=='val_seen'];unseen=[e for e in items if e['split']=='val_unseen']
            if not train or not val:continue
            programs_set=set(p for e in train for p in e['metrics']) & set(p for e in val for p in e['metrics'])
            if not programs_set:continue
            train_m={p:aggregate([e['metrics'][p] for e in train if p in e['metrics']]) for p in programs_set}
            val_m={p:aggregate([e['metrics'][p] for e in val if p in e['metrics']]) for p in programs_set}
            best,prob,reliability=fit_calibration(train_m,val_m)
            unseen_m={p:aggregate([e['metrics'][p] for e in unseen if p in e['metrics']]) for p in programs_set}
            unstable=bool(unseen_m.get(best) and (val_m[best]['auc']-.5)*(unseen_m[best]['auc']-.5)<0)
            rec={'phrase':key[0],'anchor_type':key[1],'source':key[2],'context':context,'n_train':len(train),'n_val_seen':len(val),'n_val_unseen':len(unseen),
                 'best_program':best,'train':train_m,'val_seen':val_m,'val_unseen':unseen_m,
                 'probabilities':prob,'reliability':reliability,'unstable_seen_unseen':unstable}
            calibrations[(key,context)]=rec;records.append(rec)
    # F0–F4 per evaluated row; F0 explicitly UNKNOWN for words omitted by old ontology.
    variants=defaultdict(list);variants_by_protocol=defaultdict(list)
    for e in evaluated:
        kc=((e['phrase'],e['kind'],e['source']),True);kp=((e['phrase'],'all',e['source']),False)
        cc=calibrations.get(kc);cp=calibrations.get(kp)
        if not cc or not cp:continue
        old=np.asarray(e['old_scores']) if e.get('old_scores') is not None else None
        choices={'F0':old if old is not None else np.zeros(len(next(iter(e['scores'].values())))),
                 'F1':e['scores'].get(cp['best_program']),
                 'F2':e['scores'].get(cc['best_program']),
                 'F3':combine(e['scores'],cc['probabilities']),
                 'F4':combine(e['scores'],cc['probabilities'],cc['reliability'])}
        for name,s in choices.items():
            if s is None:continue
            m=metric(s,e['target'])
            if m:
                variants[(e['split'],name)].append(m)
                variants_by_protocol[(e['split'],name,e['protocol'])].append(m)
    summary={'source_rows':len(rows),'evaluated_clause_rows':len(evaluated),'coverage':dict(coverage),
             'variants':{f'{split}:{name}':aggregate(items) for (split,name),items in variants.items()},
             'variants_by_protocol':{f'{split}:{name}:{protocol}':aggregate(items) for (split,name,protocol),items in variants_by_protocol.items()},
             'calibration_policy':'train_seen statistics; val_seen selection/probabilities/reliability; val_unseen report only'}
    (OUT/'frame_induction.json').write_text(json.dumps([r for r in records if r['best_program'].startswith(('BACK','FRONT','LEFT','RIGHT'))],indent=2))
    (OUT/'program_induction.json').write_text(json.dumps(records,indent=2))
    (OUT/'program_probabilities.json').write_text(json.dumps([{k:r[k] for k in ('phrase','anchor_type','source','context','probabilities')} for r in records],indent=2))
    (OUT/'relation_reliability.json').write_text(json.dumps([{k:r[k] for k in ('phrase','anchor_type','source','context','reliability','unstable_seen_unseen')} for r in records],indent=2))
    (OUT/'gate_f_metrics.json').write_text(json.dumps(summary,indent=2))
    (OUT/'evaluated_scores.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in evaluated))
    print(summary['variants'])
if __name__=='__main__':main()
