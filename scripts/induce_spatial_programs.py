#!/usr/bin/env python3
"""Paired diagnostic on existing relation samples; no unseen selection."""
from __future__ import annotations
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.spatial_graph.semantic_canonicalizer import hypotheses,anchor_type
from sensaturban_fpv.relation_v2 import GEOM_COLUMNS
from sensaturban_fpv.config import load_config,load_landmarks
from sensaturban_fpv.entity_geometry import group_of
OUT=ROOT/'artifacts/spatial_semantic_canonicalization';SRC=Path('/home/rental/20260922_1/Workspace/Achieved/hett-experiments/55-sensaturban-fpv-rendering/artifacts/relation_v2/relation_samples.jsonl')
I={x:i for i,x in enumerate(GEOM_COLUMNS)}
def auc(p,n):
    d=p[:,None]-n[None,:];return float((d>0).mean()+.5*(d==0).mean())
def score(g,name):
    if name=='CENTER_DISTANCE':return -g[:,I['distance']]
    if name in ('FOOTPRINT_DISTANCE','ROAD_AWARE_DISTANCE','REGION_DISTANCE','NEAR_ROAD_REGION','ROAD_ASSOCIATION','ON_ROAD_CORRIDOR'):return np.full(len(g),np.nan) # unavailable in this diagnostic schema
    if name=='BETWEEN_TERNARY':return np.full(len(g),np.nan) # second anchor unavailable in this schema
    if name.startswith(('BEHIND_','FRONT_','LEFT_','RIGHT_')):
        stem,frame=name.split('_',1)
        columns={'GLOBAL':('global_y','global_x'),'START_AGENT':('agent_ahead','agent_lateral'),
                 'CURRENT_AGENT':('agent_ahead','agent_lateral'),'ANCHOR':('anchor_along','anchor_perp')}
        if frame not in columns:return np.full(len(g),np.nan)
        forward,lateral=columns[frame];axis=forward if stem in ('BEHIND','FRONT') else lateral
        sign=-1 if stem in ('BEHIND','LEFT') else 1
        return sign*g[:,I[axis]]
    return np.full(len(g),np.nan)
def main():
    cfg=load_config();objects=load_landmarks(cfg);types={m:{int(o.id):o.object_type for o in obs.values()} for m,obs in objects.items()}
    data=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    nrows=0
    for line in SRC.open():
        r=json.loads(line);nrows+=1;g=np.asarray(r.get('annotation_rel_geom') or [],dtype=float)
        if g.ndim!=4 or not g.shape[1]:continue
        ids=r.get('annotation_anchor_ids') or [];rels=r.get('annotation_relations') or [];cids=r['candidate_ids'];t=r['target_index'];group=group_of(types.get(r['map'],{}).get(r['target_id'],''))
        negatives=[i for i,c in enumerate(cids) if i!=t and group_of(types.get(r['map'],{}).get(c,''))==group]
        if not negatives:continue
        for j,rel in enumerate(rels[:g.shape[1]]):
            if not rel or not rel.get('phrase') or j>=len(ids):continue
            valid=[k for k,v in enumerate(ids[j]) if v>=0 and k<g.shape[2]]
            if not valid:continue
            # Diagnostic anchor choice uses true target and is never exported as a deployable feature.
            k=min(valid,key=lambda k:abs(g[t,j,k,I['distance']]))
            phrase=rel['phrase'].lower();kind=anchor_type(r['annotation_phrases'][j] if j < len(r.get('annotation_phrases') or []) else '');p=g[t,j,k];n=g[negatives,j,k]
            for program in hypotheses(phrase,kind):
                ps=score(p[None,:],program);ns=score(n,program)
                if np.isfinite(ps).all() and np.isfinite(ns).all():data[(phrase,kind)][r['split']][program].append(auc(ps,ns))
    records=[]
    for (phrase,kind),splits in sorted(data.items()):
        candidates=set(splits.get('train_seen',{})) & set(splits.get('val_seen',{}))
        if not candidates:continue
        train={p:float(np.mean(splits['train_seen'][p])) for p in candidates}
        val={p:float(np.mean(splits['val_seen'][p])) for p in candidates}
        best=max(candidates,key=lambda p:(val[p],train[p]));unseen={p:float(np.mean(v)) for p,v in splits.get('val_unseen',{}).items() if v}
        nval=len(splits['val_seen'][best]);rho=max(0.,min(1.,(val[best]-.5)*2)) if nval>=20 else 0.
        # Soft weights from train and val only. Weak relations retain UNKNOWN mass.
        logits={p:2*(train[p]+val[p]-1) for p in candidates};logits['UNKNOWN']=0.
        z=sum(np.exp(x) for x in logits.values());prob={p:float(np.exp(x)/z) for p,x in logits.items()}
        records.append({'phrase':phrase,'anchor_type':kind,'n_train':len(splits['train_seen'][best]),'n_val_seen':nval,
                        'n_val_unseen':len(splits.get('val_unseen',{}).get(best,[])),
                        'best_program':best,'val_seen_auc':val[best],'val_unseen_auc':unseen.get(best),
                        'reliability':rho,'probabilities':prob,'train_auc':train,'val_seen_program_auc':val,
                        'val_unseen_program_auc':unseen})
    (OUT/'program_induction.json').write_text(json.dumps({'source_rows':nrows,'diagnostic_anchor':'GT nearest; evaluation only','records':records},indent=2))
    (OUT/'frame_induction.json').write_text(json.dumps({'records':[r for r in records if any(x in r['phrase'] for x in ('behind','front','left','right'))],
                                                      'unavailable':['FINAL_APPROACH','ROUTE','ROAD','CURRENT_AGENT'],
                                                      'reason':'existing sampled geometry lacks trajectory and road-frame features'},indent=2))
    (OUT/'spatial_lexicon.json').write_text(json.dumps(records,indent=2))
    print(nrows,len(records))
if __name__=='__main__':main()
