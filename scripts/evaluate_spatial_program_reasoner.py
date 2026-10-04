#!/usr/bin/env python3
"""Frozen val_seen-tuned, val_unseen-evaluated spatial-program benchmark."""
from __future__ import annotations
import argparse,json,math,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects
from multiagent.geometry_reasoning.anchors import anchor_record
from multiagent.geometry_reasoning.parser import parse_geometry
from multiagent.geometry_reasoning.scorer import GeometryReasoner
from multiagent.mapdata import MAP_BOUNDS
from multiagent.navigation_state import nms_topk_from_belief
from multiagent.scene_grounding.dataset import trajectory_pose
from multiagent.spatial_program.executor import ProgramExecutor
from multiagent.spatial_program.object_graph import get_object_graph
from multiagent.spatial_program.parser import parse_spatial_program

LAMBDA=(0.,.25,.5,1.,2.,4.)
VARIANTS=('P0_static_b0','P1_flat_geometry','P2_structured_old_executor','P3_axis_resolution','P4_ordinal_scope','P5_object_aware','P6_full_spatial_program')
def norm(x):return ''.join(c for c in x.casefold() if c.isalnum())
def world(map_name,row,col):
 b=MAP_BOUNDS[map_name];return (b.x_min+(col+.5)*410/30,b.y_max-(row+.5)*410/30)
def metric(dist,scores):
 order=np.argsort(-np.asarray(scores));d=np.asarray(dist)[order];closest=int(np.argmin(dist));rank=int(np.where(order==closest)[0][0])+1;r={'top1_distance_m':float(d[0]),'oracle_distance@8_m':float(d[:8].min()),'gt_candidate_rank':rank}
 for k in (1,4,8,16):
  for radius in (20,40):r[f'recall@{k}/{radius}m']=float(d[:min(k,len(d))].min()<=radius)
 return r
def aggregate(rows):
 return {'samples':len(rows),**({k:float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {})}
def anchors_for(program,map_objects,start):
 refs=[e for e in program.entities if e.role=='reference'];out=[]
 for e in refs:
  exact=[o for o in map_objects if o.name and norm(o.name)==norm(e.name)]
  pool=exact or [o for o in map_objects if norm(e.name) in norm(o.name+' '+o.object_type)]
  if pool:
   chosen=min(pool,key=lambda o:(o.position.x-start[0])**2+(o.position.y-start[1])**2);out.append(anchor_record(chosen))
 return out
def object_points(row,objects,radius=30.):
 graph=get_object_graph(row['map'],objects);target=next((e for e in row['program'].entities if e.role=='target'),None);kind=target.type.casefold() if target else 'region';all_points=[];parents=[]
 for i,xy in enumerate(row['xy']):
  nearby=graph.nearby(xy,radius,kind);points=[n.centroid for n,_ in nearby] or [xy]
  for p in points:all_points.append(p);parents.append(i)
 return all_points,parents
def execute_object_aware(row,objects,mode):
 points,parents=object_points(row,objects);prior=np.asarray([row['b0'][p]/parents.count(p) for p in parents]);out,debug=ProgramExecutor().execute(row['program'],points,prior,row['anchors'],row['start'][:2],row['start'].yaw,objects)
 agg=np.zeros(16)
 for i in range(16):
  vals=out[np.asarray(parents)==i]
  if not len(vals):continue
  agg[i]=vals.max() if mode=='max' else math.log(np.exp(vals*20).sum())/20
 return agg/max(agg.sum(),1e-12),debug
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--b0-cache-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert 'test_unseen' not in ' '.join(map(str,vars(a).values()));a.output.mkdir(parents=True,exist_ok=True);started=time.time()
 objects=get_city_refer_objects(a.data_root/'cityrefer/objects.json',a.data_root/'cityrefer/processed_descriptions.json');prepared={};counts={}
 for split in ('val_seen','val_unseen'):
  episodes=json.loads((a.data_root/'processed_citynav'/f'citynav_{split}.json').read_text());beliefs=torch.load(a.b0_cache_root/f'{split}_b0_eall.pt',map_location='cpu',weights_only=True);rows=[];counter=Counter()
  for index,(e,belief) in enumerate(zip(episodes,beliefs)):
   m=f"{e['area']}_block_{e['block']}";o=objects[m][int(e['object_ids'][0])];ann=int(e['ann_ids'][0]);instruction=o.descriptions[ann];refs=o.processed_descriptions[ann].landmarks;pgr=parse_spatial_program(instruction,refs);flat=parse_geometry(instruction,refs);start=trajectory_pose(e['trajectory'][0]);anchors=anchors_for(pgr,objects[m].values(),start)
   peaks=nms_topk_from_belief(belief,top_k=16,kernel_size=3);xy=[world(m,x.row,x.col) for x in peaks];b0=np.asarray([x.probability for x in peaks]);goal=np.asarray(e['target_positions'][-1][:2]);dist=np.linalg.norm(np.asarray(xy)-goal,axis=1)
   rows.append({'index':index,'map':m,'instruction':instruction,'program':pgr,'flat':flat,'start':start,'anchors':anchors,'xy':xy,'b0':b0,'dist':dist,'goal':goal})
   counter['instructions']+=1;counter['with_program']+=bool(pgr.clauses);counter[f'relations_{min(len(pgr.clauses),3)}'+('+' if len(pgr.clauses)>=3 else '')]+=1
   for c in pgr.clauses:counter[c.relation]+=1
  prepared[split]=rows;counts[split]=dict(counter)
 flat_reasoner=GeometryReasoner()
 def raw_scores(row,variant):
  if variant=='P0_static_b0':return np.log(np.maximum(row['b0'],1e-12)),{}
  if variant=='P1_flat_geometry':
   # Previous implementation retained exactly for paired comparison.
   groups=[{'query':a['name'],'matches':[a]} for a in row['anchors']];d=flat_reasoner.score(row['flat'],groups,row['xy'],row['start'][:2],row['start'].yaw,None,True);return np.asarray([x['total'] for x in d]),{'flat':d}
  if variant in ('P5_object_aware','P6_full_spatial_program'):
   probability,debug=execute_object_aware(row,list(objects[row['map']].values()),'max' if variant=='P5_object_aware' else 'logsumexp');return np.log(np.maximum(probability,1e-12))-np.log(np.maximum(row['b0'],1e-12)),debug
  enabled=() if variant=='P2_structured_old_executor' else ('axis',) if variant=='P3_axis_resolution' else ('axis','ordinal')
  probability,debug=ProgramExecutor().execute(row['program'],row['xy'],row['b0'],row['anchors'],row['start'][:2],row['start'].yaw,list(objects[row['map']].values()),enabled);return np.log(np.maximum(probability,1e-12))-np.log(np.maximum(row['b0'],1e-12)),debug
 # Geometry is deterministic: compute it once, then tune lambda on cached signals.
 signal_cache={s:{v:[] for v in VARIANTS} for s in prepared};trace_cache={s:[] for s in prepared}
 for split,rows in prepared.items():
  for row in rows:
   for variant in VARIANTS:
    signal,trace=raw_scores(row,variant);signal_cache[split][variant].append(signal)
    if variant=='P6_full_spatial_program':trace_cache[split].append(trace)
 def evaluate(split,variant,lamb,subset=None):
  out=[]
  for row,signal in zip(prepared[split],signal_cache[split][variant]):
   if subset and not subset(row):continue
   final=np.log(np.maximum(row['b0'],1e-12))+lamb*signal if variant!='P0_static_b0' else signal;out.append(metric(row['dist'],final))
  return aggregate(out)
 tuned={};variants={}
 for v in VARIANTS:
  if v=='P0_static_b0':w=0.
  else:w=max(LAMBDA,key=lambda x:evaluate('val_seen',v,x)['recall@1/20m']+evaluate('val_seen',v,x)['recall@4/20m'])
  tuned[v]=w;variants[v]={'lambda':w,'val_seen':evaluate('val_seen',v,w),'val_unseen':evaluate('val_unseen',v,w),
                         'fixed_lambda_1_val_seen':evaluate('val_seen',v,1.),'fixed_lambda_1_val_unseen':evaluate('val_unseen',v,1.)}
 relation={};rels=('left_of','right_of','front_of','behind','near','past','before','after','between','ordinal')
 for rel in rels:
  subset=lambda row,r=rel:any(c.relation==r for c in row['program'].clauses)
  relation[rel]={s:{'P0':evaluate(s,'P0_static_b0',0,subset),'P6_tuned':evaluate(s,'P6_full_spatial_program',tuned['P6_full_spatial_program'],subset),'P6_fixed_lambda_1':evaluate(s,'P6_full_spatial_program',1.,subset)} for s in ('val_seen','val_unseen')}
 complexity={}
 for label,predicate in [('1_relation',lambda n:n==1),('2_relations',lambda n:n==2),('3+_relations',lambda n:n>=3)]:
  subset=lambda row,pred=predicate:pred(len(row['program'].clauses));complexity[label]={s:{'P1':evaluate(s,'P1_flat_geometry',tuned['P1_flat_geometry'],subset),'P6_tuned':evaluate(s,'P6_full_spatial_program',tuned['P6_full_spatial_program'],subset),'P6_fixed_lambda_1':evaluate(s,'P6_full_spatial_program',1.,subset)} for s in ('val_seen','val_unseen')}
 anchor_types={}
 for kind in ('church','library','college','school','parking lot','road','building','house','bridge'):
  subset=lambda row,k=kind:any(k in (e.type+' '+e.name).casefold() for e in row['program'].entities if e.role=='reference');anchor_types[kind]={'val_unseen':{'P0':evaluate('val_unseen','P0_static_b0',0,subset),'P6':evaluate('val_unseen','P6_full_spatial_program',tuned['P6_full_spatial_program'],subset)}}
 # Failure classes are mutually exclusive and use GT only after inference.
 failure=Counter();debug=[]
 for split in ('val_seen','val_unseen'):
  for row,signal,trace in zip(prepared[split],signal_cache[split]['P6_full_spatial_program'],trace_cache[split]):
   m=metric(row['dist'],np.log(np.maximum(row['b0'],1e-12))+tuned['P6_full_spatial_program']*signal)
   if row['dist'].min()>20:reason='candidate_pool_miss'
   elif not row['program'].clauses:reason='relation_detection_or_no_explicit_relation'
   elif any(c.object is None and c.relation!='ordinal' for c in row['program'].clauses):reason='relation_binding_error'
   elif not row['anchors']:reason='anchor_grounding_failure'
   elif trace.get('axis',{}).get('confidence',0)<.5:reason='axis_resolution_low_confidence'
   elif any(c.relation=='ordinal' for c in row['program'].clauses) and m['gt_candidate_rank']>4:reason='ordinal_scope_or_ordering_error'
   elif m['gt_candidate_rank']>4:reason='correct_program_but_wrong_reranking'
   else:reason='success_top4'
   failure[f'{split}:{reason}']+=1
   if len(debug)<200 and ((split=='val_seen' and sum(k.startswith('val_seen:') for k in [d['split']+':' for d in debug])<100) or split=='val_unseen'):
    debug.append({'split':split,'episode_index':row['index'],'instruction':row['instruction'],'program':row['program'].to_dict(),'anchors':row['anchors'],'start_pose':list(row['start']),'candidates':[{'xy':list(map(float,xy)),'b0':float(b),'distance_to_gt_eval_only':float(d)} for xy,b,d in zip(row['xy'],row['b0'],row['dist'])],'trace':trace,'metrics':m,'failure':reason})
 report={'protocol':{'rgb_used':False,'controller_used':False,'rollout_used':False,'test_unseen_read':False,'gt_inference_input':False,'tuning_split':'val_seen','frozen_eval_split':'val_unseen'},'counts':counts,'variants':variants,'relation_level':relation,'multi_relation':complexity,'anchor_types':anchor_types,'failure_taxonomy':dict(failure),'runtime_seconds':time.time()-started,'executor_parameters':0,'gold_program':{'status':'not_evaluated_without_human_review','reason':'semi-automatic review queue is not human gold; reporting it as P7 would be circular'}}
 (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'debug_records.json').write_text(json.dumps(debug,indent=2)+'\n');print(json.dumps({'runtime_seconds':report['runtime_seconds'],'P0':variants['P0_static_b0']['val_unseen'],'P6':variants['P6_full_spatial_program']['val_unseen']},indent=2))
if __name__=='__main__':main()
