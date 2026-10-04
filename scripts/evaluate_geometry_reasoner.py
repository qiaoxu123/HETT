#!/usr/bin/env python3
"""Evaluate static/dynamic analytic geometry without RGB or rollout."""
from __future__ import annotations
import argparse,json,math,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects
from multiagent.geometry_reasoning.anchors import resolve_anchors
from multiagent.geometry_reasoning.parser import parse_geometry
from multiagent.geometry_reasoning.scorer import GeometryReasoner
from multiagent.mapdata import MAP_BOUNDS
from multiagent.navigation_state import nms_topk_from_belief
from multiagent.scene_grounding.dataset import trajectory_pose

RELATIONS={"direction":{"left_of","right_of","front_of","behind"},"near":{"near","nearest","farthest"},"ordinal":{"ordinal"},"sequence":{"before","after","past"},"between":{"between"}}
VARIANTS={"G0_b0":(set(),False,"start",True),"G1_anchor_distance":(set(),True,"start",True),
 "G2_left_right":({"left_of","right_of"},False,"start",True),"G3_ordinal":({"ordinal"},False,"start",True),
 "G4_sequence":(RELATIONS["sequence"],False,"start",True),"G5_all_geometry":(None,False,"orientation",True),
 "G6_geometry_only":(None,True,"start",False),"G7_b0_geometry_start":(None,True,"start",True)}
LAMBDAS=(0.,.25,.5,1.,2.,4.,8.)

def world(map_name,row,col,grid=30):
 b=MAP_BOUNDS[map_name];return (b.x_min+(col+.5)*410/grid,b.y_max-(row+.5)*410/grid)
def angle_score(candidate,goal,anchor):
 a=np.asarray(anchor); u=np.asarray(candidate)-a;v=np.asarray(goal)-a
 if np.linalg.norm(u)<1e-6 or np.linalg.norm(v)<1e-6:return 0.
 cosine=float(np.clip((u@v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1));return 1-math.acos(cosine)/math.pi
def record_metric(distances,scores,closest_original):
 order=np.argsort(-np.asarray(scores)); d=np.asarray(distances)[order]; rank=int(np.where(order==closest_original)[0][0])+1
 result={"top1_distance_m":float(d[0]),"oracle_distance@8_m":float(d[:8].min()),"gt_candidate_rank":rank}
 for k in (1,4,8,16):
  for radius in (20,40):result[f"recall@{k}/{radius}m"]=float(d[:min(k,len(d))].min()<=radius)
 return result
def aggregate(rows):
 if not rows:return {"samples":0}
 return {"samples":len(rows)}|{k:float(np.mean([r[k] for r in rows])) for k in rows[0]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--b0-cache-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if 'test_unseen' in ' '.join(map(str,vars(a).values())):raise ValueError('test_unseen forbidden')
 objects=get_city_refer_objects(a.data_root/'cityrefer/objects.json',a.data_root/'cityrefer/processed_descriptions.json'); reasoner=GeometryReasoner(); started=time.time()
 programs={}; coverage={}; prepared={}; relation_counts={}
 for split in ('train_seen','val_seen','val_unseen'):
  episodes=json.loads((a.data_root/'processed_citynav'/f'citynav_{split}.json').read_text()); counts=Counter(); plist=[]
  for idx,e in enumerate(episodes):
   m=f"{e['area']}_block_{e['block']}"; obj=objects[m][int(e['object_ids'][0])]; ann=int(e['ann_ids'][0]); processed=obj.processed_descriptions[ann]
   program=parse_geometry(obj.descriptions[ann],processed.landmarks);plist.append(program);counts['instructions']+=1;counts['with_constraint']+=bool(program.constraints);counts['with_anchor']+=bool(program.anchors)
   for c in program.constraints:counts[c['type']]+=1;counts['supported']+=c['supported']
  programs[split]=plist;coverage[split]=dict(counts)
 for split in ('val_seen','val_unseen'):
  episodes=json.loads((a.data_root/'processed_citynav'/f'citynav_{split}.json').read_text());beliefs=torch.load(a.b0_cache_root/f'{split}_b0_eall.pt',map_location='cpu',weights_only=True); rows=[]
  for idx,(e,belief,program) in enumerate(zip(episodes,beliefs,programs[split])):
   m=f"{e['area']}_block_{e['block']}"; peaks=nms_topk_from_belief(belief,top_k=16,kernel_size=3); xy=[world(m,x.row,x.col) for x in peaks]; b0=np.asarray([x.probability for x in peaks]); goal=np.asarray(e['target_positions'][-1][:2],float); distances=np.linalg.norm(np.asarray(xy)-goal,axis=1); closest=int(np.argmin(distances)); start=trajectory_pose(e['trajectory'][0]); anchors=resolve_anchors(program,objects[m].values())
   rows.append({'episode':idx,'map':m,'program':program,'anchors':anchors,'xy':xy,'b0':b0,'goal':goal,'distances':distances,'closest':closest,'trajectory':e['trajectory'],'start':start})
  prepared[split]=rows
 def scores_for(row,variant,progress=0.,anchor_mode='true'):
  filt,anchor_distance,frame,use_b0=VARIANTS[variant];anchors=row['anchors']
  if anchor_mode!='true':
   pool=[o for o in objects[row['map']].values() if o.name]
   if anchor_mode=='none':anchors=[]
   elif pool:
    chosen=pool[(row['episode']+(1 if anchor_mode=='shuffle' else 7))%len(pool)]
    anchors=[{'query':'control','matches':[__import__('multiagent.geometry_reasoning.anchors',fromlist=['anchor_record']).anchor_record(chosen)]}]
  if progress>0:
   index=min(round(progress*(len(row['trajectory'])-1)),len(row['trajectory'])-1);pose=trajectory_pose(row['trajectory'][index])
  else:pose=row['start']
  if frame=='orientation' and anchors and anchors[0]['matches']:
   c=anchors[0]['matches'][0]['centroid'];pose_xy=c;yaw=None
  else:pose_xy=pose[:2];yaw=pose.yaw
  detail=reasoner.score(row['program'],anchors,row['xy'],pose_xy,yaw,filt,anchor_distance);geom=np.asarray([x['total'] for x in detail]);return geom,detail,use_b0
 def evaluate(split,variant,lamb,progress=0.,anchor_mode='true',pool_k=16):
  output=[]
  for row in prepared[split]:
   geom,detail,use_b0=scores_for(row,variant,progress,anchor_mode); n=pool_k
   final=(np.log(np.maximum(row['b0'][:n],1e-12)) if use_b0 else np.zeros(n))+lamb*geom[:n]
   metric=record_metric(row['distances'][:n],final,int(np.argmin(row['distances'][:n])))
   valid=[d['valid'] for d in detail[:n]];metric['valid_geometry']=float(any(valid));metric['geometry_margin']=float(geom[row['closest']]-np.max(np.delete(geom,row['closest']))) if len(geom)>1 else 0.;output.append(metric)
  return aggregate(output)
 tuned={};results={}
 for variant in VARIANTS:
  weight=0. if variant=='G0_b0' else 1. if variant=='G6_geometry_only' else max(LAMBDAS,key=lambda w:evaluate('val_seen',variant,w)['recall@1/20m']+evaluate('val_seen',variant,w)['recall@4/20m'])
  tuned[variant]=weight;results[variant]={s:evaluate(s,variant,weight) for s in ('val_seen','val_unseen')};results[variant]['lambda']=weight
 single={}
 for name,filt in RELATIONS.items():
  key='G7_b0_geometry_start'; old=VARIANTS[key];VARIANTS[key]=(filt,False,'start',True);w=max(LAMBDAS,key=lambda x:evaluate('val_seen',key,x)['recall@1/20m']);single[name]={'lambda':w,'val_seen':evaluate('val_seen',key,w),'val_unseen':evaluate('val_unseen',key,w)};VARIANTS[key]=old
 controls={mode:{s:evaluate(s,'G7_b0_geometry_start',tuned['G7_b0_geometry_start'],anchor_mode=mode) for s in ('val_seen','val_unseen')} for mode in ('true','shuffle','random','none')}
 temporal={}
 for progress in (0,.25,.5,.75,1.):temporal[str(progress)]={s:evaluate(s,'G7_b0_geometry_start',tuned['G7_b0_geometry_start'],progress) for s in ('val_seen','val_unseen')}
 pools={str(k):{s:evaluate(s,'G7_b0_geometry_start',tuned['G7_b0_geometry_start'],pool_k=k) for s in ('val_seen','val_unseen')} for k in (4,8,16)}
 # Privileged relation-signature Oracles: GT is used only here and in metrics.
 oracle_rows={s:defaultdict(list) for s in ('val_seen','val_unseen')};failure=Counter();distance_rows=defaultdict(list)
 for split,rows in prepared.items():
  for row in rows:
   resolved=[m for g in row['anchors'] for m in g['matches']]; anchor=np.asarray(resolved[0]['centroid']) if resolved else None
   base=np.log(np.maximum(row['b0'],1e-12)); signatures={}
   if anchor is not None:
    gd=np.linalg.norm(row['goal']-anchor);cd=np.linalg.norm(np.asarray(row['xy'])-anchor,axis=1);signatures['anchor_distance_signature']=np.exp(-np.abs(cd-gd)/20)
    signatures['bearing_signature']=np.asarray([angle_score(x,row['goal'],anchor) for x in row['xy']])
    signatures['all_geometry_signature']=(signatures['anchor_distance_signature']+signatures['bearing_signature'])/2
    pose=row['start'];
    for rel_name,rel_filter in RELATIONS.items():
     detail=reasoner.score(row['program'],row['anchors'],row['xy']+[row['goal'].tolist()],pose[:2],pose.yaw,rel_filter,False)
     if detail[-1]['valid']:
      candidate=np.asarray([x['total'] for x in detail[:-1]]); signatures[f'{rel_name}_relation']=np.exp(-np.abs(candidate-detail[-1]['total'])/.2)
   for name,values in signatures.items():oracle_rows[split][name].append(record_metric(row['distances'],base+16*values,row['closest']))
   if split=='val_unseen':
    geom,detail,_=scores_for(row,'G7_b0_geometry_start');final=np.log(np.maximum(row['b0'],1e-12))+tuned['G7_b0_geometry_start']*geom;metric=record_metric(row['distances'],final,row['closest'])
    if row['distances'].min()>20:failure['candidate_pool_miss']+=1
    elif not row['program'].constraints:failure['parser_no_constraint']+=1
    elif not resolved:failure['anchor_unresolved']+=1
    elif not any(x['valid'] for x in detail):failure['invalid_geometry']+=1
    elif metric['gt_candidate_rank']>4 and sum(c.get('supported',False) for c in row['program'].constraints)>1:failure['rule_conflict']+=1
    elif metric['gt_candidate_rank']>4:failure['ranking_error']+=1
    else:failure['success_top4']+=1
    if anchor is not None:
     d=float(np.linalg.norm(row['goal']-anchor));bucket='0-20m' if d<20 else '20-40m' if d<40 else '40-80m' if d<80 else '80m+';distance_rows[bucket].append(metric)
 oracle_names=sorted(set().union(*(oracle_rows[s].keys() for s in oracle_rows)))
 oracles={name:{s:aggregate(oracle_rows[s][name]) for s in ('val_seen','val_unseen')} for name in oracle_names}
 report={'coverage':coverage,'variants':results,'single_relations':single,'anchor_controls':controls,'temporal':temporal,'candidate_pools':pools,'distance_buckets':{k:aggregate(v) for k,v in distance_rows.items()},'failure_taxonomy':dict(failure),'oracles':oracles,'parameters':0,'seconds':time.time()-started,'test_unseen_read':False,'rgb_used':False,'gt_input_used':False}
 (a.output/'metrics.json').write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps({'seconds':report['seconds'],'G7':results['G7_b0_geometry_start']['val_unseen'],'oracle':oracles['all_geometry_signature']['val_unseen']},indent=2))
if __name__=='__main__':main()
