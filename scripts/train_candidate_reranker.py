#!/usr/bin/env python3
"""Cache train_seen policy states, train only a residual head for at most 2 epochs."""
import os,sys,json,time,hashlib,argparse,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_candidate_controller import NavCMTAgent,CityNavBatch,parse_args,RUN
from multiagent.models.candidate_reranker import CandidateReranker,FEATURE_KEYS,install_candidate_features
from torch.utils.data import DataLoader

class Writer:
 def __init__(self,path,limit):
  self.path=path;path.mkdir(parents=True);self.limit=limit;self.parts=defaultdict(list);self.ids=[];self.count=0;self.total=0;self.index=0
 def __call__(self,features,distances,mask,ids,t):
  valid=mask.detach().cpu().tolist()
  if not any(valid):return
  for k in FEATURE_KEYS:
   v=features[k][mask].detach().cpu();self.parts[k].append(v if v.dtype==torch.bool else v.half())
  self.parts['distances'].append(distances[mask].detach().cpu().float())
  self.ids.extend([{'episode_id':list(eid),'step':t} for eid,ok in zip(ids,valid) if ok]);n=sum(valid);self.count+=n;self.total+=n
  if self.count>=self.limit:self.flush()
 def flush(self):
  if not self.count:return
  data={k:torch.cat(v) for k,v in self.parts.items()};data['sample_ids']=self.ids
  torch.save(data,self.path/f'{self.index:05d}.pt');self.index+=1;self.parts=defaultdict(list);self.ids=[];self.count=0

def main():
 cli=argparse.ArgumentParser();cli.add_argument('--config',type=Path,default=ROOT/'configs/diagnostics/train_reranker.json');cli.add_argument('--output',type=Path,default=ROOT/'artifacts/reranker_training');cli.add_argument('--cache-only',action='store_true');opt=cli.parse_args()
 cfg=json.loads(opt.config.read_text());out=opt.output.resolve()
 if not 1<=cfg['epochs']<=2:raise ValueError('training budget is capped at 2 epochs')
 out.mkdir(parents=True,exist_ok=True)
 if (out/'epoch1.pt').exists():raise FileExistsError('refuse to overwrite trained checkpoints')
 torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
 checkpoint=RUN/'best_val_unseen';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 manifest={'config':cfg,'base_checkpoint':str(checkpoint),'base_checkpoint_sha256':digest,'train_split':'train_seen','training_budget_epochs':cfg['epochs'],'frozen_backbone':True,'started_at':time.time(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 sys.argv=['reranker_cache'];args=parse_args()
 for k,v in json.loads((RUN/'training_args.json').read_text()).items():setattr(args,k,v)
 args.mode='eval';args.resume_optimizer=False;args.batch_size=cfg['collection_batch_size'];args.heatmap_execution='waypoint';args.diagnostic={}
 agent=NavCMTAgent(args,allow_ngpus=False,rank=0);agent.load(str(checkpoint))
 for model in [agent.lang_model,agent.vision_model,agent.vln_model]:
  model.eval()
  for p in model.parameters():p.requires_grad_(False)
 cache=out/'cache'
 if not (out/'cache_complete.json').exists():
  if cache.exists():raise RuntimeError('incomplete cache exists; use a fresh output')
  env=CityNavBatch('train_seen',args,batch_size=args.batch_size,seed=cfg['seed'],rank=0,world_size=1)
  if cfg.get('max_train_episodes'):env.data=env.data[:cfg['max_train_episodes']]
  # Last legacy batch pads with existing episodes; exclude duplicate sample keys below when training.
  writer=Writer(cache,cfg['shard_samples']);install_candidate_features(agent)
  agent.candidate_feature_collector=writer;agent.candidate_collect_steps=set(cfg['collect_steps']);agent.env=env;agent.logs=defaultdict(list)
  start=time.time();agent.test(DataLoader(env,batch_size=1),env_name='train_seen',feedback='student');writer.flush()
  (out/'cache_complete.json').write_text(json.dumps({'episodes':len(env.data),'samples':writer.total,'shards':writer.index,'seconds':time.time()-start})+'\n')
  agent.candidate_feature_collector=None
 if getattr(agent,'_candidate_feature_hook',None) is not None:agent._candidate_feature_hook.remove()
 del agent
 import gc
 gc.collect();torch.cuda.empty_cache()
 if opt.cache_only:return
 torch.manual_seed(cfg['seed'])
 model=CandidateReranker().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 shards=sorted(cache.glob('*.pt'))
 for epoch in range(1,cfg['epochs']+1):
  model.train();rng=np.random.default_rng(cfg['seed']+epoch);order=rng.permutation(len(shards));seen=set();tot_loss=0.;total=0;correct=0;start=time.time()
  for si in order:
   data=torch.load(shards[si],map_location='cpu');eligible=[]
   for j,item in enumerate(data['sample_ids']):
    key=(*item['episode_id'],item['step'])
    if key not in seen:eligible.append(j);seen.add(key)
   perm=rng.permutation(eligible)
   for off in range(0,len(perm),cfg['batch_size']):
    ii=torch.as_tensor(perm[off:off+cfg['batch_size']],dtype=torch.long)
    f={k:data[k][ii].cuda().bool() if k=='landmark_valid' else data[k][ii].cuda().float() for k in FEATURE_KEYS}
    d=data['distances'][ii].cuda();scores=model(f)
    logp=scores.log_softmax(-1);soft=(-d/cfg['temperature_m']).softmax(-1)
    loss=cfg['soft_loss_weight']*(-soft*logp).sum(-1).mean()+cfg['hard_loss_weight']*torch.nn.functional.cross_entropy(scores,d.argmin(-1))
    if not torch.isfinite(loss):raise RuntimeError('nonfinite reranker loss')
    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
    n=len(ii);total+=n;tot_loss+=float(loss)*n;correct+=int((scores.argmax(-1)==d.argmin(-1)).sum())
  path=out/f'epoch{epoch}.pt';torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'base_checkpoint_sha256':digest,'config':cfg},path)
  record={'epoch':epoch,'samples':total,'loss':tot_loss/total,'nearest_accuracy':correct/total,'seconds':time.time()-start,'checkpoint':str(path)}
  with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print('TRAIN_EPOCH_COMPLETE',json.dumps(record),flush=True)
  evaluation={'splits':['val_unseen','val_seen'],'variants':{'baseline':{'record':True,'execution':'waypoint'},f'reranker_epoch{epoch}':{'record':True,'execution':'waypoint','reranker_checkpoint':str(path)}}}
  config=out/f'eval_epoch{epoch}.json';config.write_text(json.dumps(evaluation,indent=2)+'\n')
  evalout=out/f'eval_epoch{epoch}'
  cmd=[sys.executable,str(ROOT/'scripts/diagnose_candidate_controller.py'),'--config',str(config),'--output',str(evalout)]
  with (out/f'eval_epoch{epoch}.log').open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 # Freeze selection on val_unseen only. Full-split eval is conditional on a real proxy gain.
 candidates=[]
 for epoch in range(1,cfg['epochs']+1):
  rows=json.loads((out/f'eval_epoch{epoch}/results.json').read_text());base=next(r for r in rows if r['split']=='val_unseen' and r['variant']=='baseline');r=next(r for r in rows if r['split']=='val_unseen' and r['variant']!='baseline');candidates.append((r['sr'],r['spl'],-r['ne'],epoch,r,base))
 best=max(candidates,key=lambda x:x[:4]);improved=best[4]['sr']>best[5]['sr']
 selection={'selected_epoch':best[3],'proxy_improved':improved,'selected_unseen':best[4],'baseline_unseen':best[5]}
 (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 if improved:
  cfgpath=out/f'eval_epoch{best[3]}.json';fullout=out/'full_validation'
  with (out/'full_validation.log').open('w') as f:subprocess.run([sys.executable,str(ROOT/'scripts/diagnose_candidate_controller.py'),'--config',str(cfgpath),'--output',str(fullout),'--full'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 (out/'COMPLETE.json').write_text(json.dumps({'epochs_trained':cfg['epochs'],'proxy_improved':improved,'time':time.time()})+'\n')
if __name__=='__main__':main()
