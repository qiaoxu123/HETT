#!/usr/bin/env python3
"""Frozen checkpoint paired diagnostics. Oracle variants are privileged only."""
import os,sys,json,time,hashlib,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

def json_default(value):
 if isinstance(value,np.generic): return value.item()
 if isinstance(value,np.ndarray): return value.tolist()
 raise TypeError(type(value).__name__)
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/candidate_controller'
SOURCE=ROOT.parent/'hett-heatmap-sbf-relative-geometry-top20'
RUN=SOURCE/'checkpoints/heatmap_relative_geometry_top20_688e981_b8_s0_2e_20261008'
SBF=ROOT.parent/'SBFNav-CityNav-Reproduction'
os.chdir(ROOT/'multiagent');sys.path[:0]=[str(ROOT/'multiagent'),str(ROOT)]
original_load=torch.load
def trusted_load(*a,**kw):
 kw.setdefault('weights_only',False);return original_load(*a,**kw)
torch.load=trusted_load
from agent import NavCMTAgent
from env import CityNavBatch
from parser import parse_args
from torch.utils.data import DataLoader

def main():
 global OUT
 cli=argparse.ArgumentParser()
 cli.add_argument('--config',type=Path,default=ROOT/'configs/diagnostics/candidate_controller.json')
 cli.add_argument('--output',type=Path,default=OUT)
 cli.add_argument('--full',action='store_true')
 options=cli.parse_args();OUT=options.output.resolve()
 cfg=json.loads(options.config.read_text())
 sys.argv=['diagnostic'];args=parse_args()
 for k,v in json.loads((RUN/'training_args.json').read_text()).items():setattr(args,k,v)
 args.mode='eval';args.resume_optimizer=False;args.batch_size=8
 OUT.mkdir(parents=True,exist_ok=True)
 checkpoint=RUN/'best_val_unseen'
 import subprocess,datetime
 metadata={'started_at':datetime.datetime.now().astimezone().isoformat(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'config':cfg,'full_split':options.full,'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'multiagent/agent.py',ROOT/'multiagent/heatmap_execution.py',Path(__file__)]}}
 (OUT/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
 torch.manual_seed(0);np.random.seed(0)
 agent=NavCMTAgent(args,allow_ngpus=False,rank=0)
 agent.load(str(RUN/'best_val_unseen'))
 weights={k:v.detach().clone() for k,v in agent.vln_model.named_parameters() if k.endswith(('language_gate','geometry_gate','multi_landmark_gate'))}
 rows=[]
 for split in cfg['splits']:
  env=CityNavBatch(split,args,batch_size=8,seed=0,rank=0,world_size=1)
  wanted=json.loads((SBF/'artifacts/sbfnav_hett_ablation/episode_ids'/f'{split}.json').read_text())
  if options.full: wanted=[list(e.id) for e in env.data]
  byid={tuple(e.id):e for e in env.data};env.data=[byid[tuple(i)] for i in wanted]
  base=None
  for name,variant in cfg['variants'].items():
   path=OUT/f'{split}_{name}.json'
   if path.exists():
    r=json.loads(path.read_text());rows.append(r['summary'])
    if name=='baseline':base=r['episodes']
    continue
   args.diagnostic=variant
   args.heatmap_execution=variant.get('execution','two_stage')
   args.stage1_switch_dist=variant.get('switch_dist',25.0)
   with torch.no_grad():
    for k,v in agent.vln_model.named_parameters():
     if k in weights:v.copy_(torch.zeros_like(v) if variant.get('gates_off') or any(k.endswith(g) for g in variant.get('disable_gates',[])) else weights[k])
   agent.env=env;agent.logs=defaultdict(list)
   print('START',split,name,flush=True);start=time.time()
   agent.test(DataLoader(env,batch_size=1),env_name=split,feedback='student')
   predictions=agent.get_results();summary,_=env.eval_metrics(predictions)
   summary={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in summary.items()}
   episodes=[]
   for eid in wanted:
    p=predictions[tuple(eid)];pathxy=[q.xy for q in p['trajectory']];goal=p['goal']
    m=env._eval_item([q.xy for q in p['gt_trajectory']],pathxy,goal)
    ds=[q.dist_to(goal) for q in pathxy]
    episodes.append({'episode_id':eid,'success':m['success'],'osr':m['oracle_success'],'spl':m['spl'],'ne':m['ne'],
     'path_xy':[list(q) for q in pathxy],'goal_xy':list(goal),'entered_then_left':min(ds)<=20 and ds[-1]>20,
     'stop_reason':p.get('stop_reason',[]),'stage2_entry_distances':p.get('stage2_entry_distances',[]),'diagnostics':p['diagnostics']})
   if name=='baseline':base=episodes
   dif=np.array([r['success']-b['success'] for r,b in zip(episodes,base)])
   boot=np.random.default_rng(0).choice(dif,(10000,len(dif)),replace=True).mean(1)*100
   first=[r['diagnostics'][0] for r in episodes];steps=[d for r in episodes for d in r['diagnostics']]
   summary.update(variant=name,split=split,episodes=len(episodes),seconds=time.time()-start,
    delta_sr_pp=float(dif.mean()*100),delta_sr_ci95=np.quantile(boot,[.025,.975]).tolist(),
    gained=int((dif>0).sum()),lost=int((dif<0).sum()),entered_then_left=sum(r['entered_then_left'] for r in episodes),
    privileged=bool(variant.get('oracle_candidate') or variant.get('oracle_stop')))
   for prefix,data in [('initial',first),('allsteps',steps)]:
    for key in ['top1_distance','top16_distance','top20_distance','refined_distance']:
     summary[prefix+'_'+key+'_hit20']=float(np.mean([d[key]<=20 for d in data])*100)
     summary[prefix+'_'+key+'_mean']=float(np.mean([d[key] for d in data]))
   entries=[d for r in episodes for d in r['stage2_entry_distances']]
   summary['stage2_entries']=len(entries)
   summary['stage2_entry_mean_m']=float(np.mean(entries)) if entries else None
   summary['stage2_entry_over40_rate']=float(np.mean(np.array(entries)>40)) if entries else None
   path.write_text(json.dumps({'summary':summary,'episodes':episodes},indent=2,default=json_default)+'\n')
   rows.append(summary);(OUT/'results.json').write_text(json.dumps(rows,indent=2,default=json_default)+'\n')
   print('DONE',json.dumps(summary,default=json_default),flush=True)
 (OUT/'COMPLETE.json').write_text(json.dumps({'runs':len(rows),'time':time.time()})+'\n')
if __name__=='__main__':main()
