#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.rsrefseg2_grounding.io import GroundingDataset,collate
from multiagent.rsrefseg2_grounding.metrics import aggregate,frame_result,ranking_loss,soft_localization_loss
from multiagent.rsrefseg2_grounding.official_adapter import OfficialCoarseGrounder,configure_trainable

def contrastive(output,target):
 mask=F.interpolate(target.float(),size=output['refined_features'].shape[-2:],mode='area');visual=(output['refined_features']*mask).flatten(2).sum(-1)/mask.flatten(2).sum(-1).clamp_min(1e-6);visual=F.normalize(visual.float(),dim=-1);text=F.normalize(output['text_pool'].float(),dim=-1);return F.cross_entropy(visual@text.T/.07,torch.arange(len(visual),device=visual.device))
@torch.no_grad()
def validate(model,loader,device):
 model.eval();rows=[]
 for batch in loader:
  with torch.autocast(device.type,dtype=torch.bfloat16):out=model(batch['images'].to(device,dtype=torch.bfloat16),batch['texts'])
  for i,(row,payload) in enumerate(zip(batch['rows'],batch['payloads'])):rows.append(frame_result(row,out['coarse_logits'][i].float().cpu(),payload))
 return aggregate(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--official-root',type=Path,required=True);p.add_argument('--official-checkpoint',type=Path,required=True);p.add_argument('--mode',choices=('prompter','vision_lora'),required=True);p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accumulate',type=int,default=16);p.add_argument('--limit-train',type=int,default=0);p.add_argument('--limit-val',type=int,default=0);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
 if 'test_unseen' in ' '.join(map(str,vars(a).values())):raise ValueError('test_unseen forbidden')
 torch.manual_seed(a.seed);a.output.mkdir(parents=True,exist_ok=True);device=torch.device('cuda');model=OfficialCoarseGrounder(a.official_root);loaded=model.load_official_checkpoint(a.official_checkpoint);trainable=configure_trainable(model,a.mode);model.to(device,dtype=torch.bfloat16)
 train=DataLoader(GroundingDataset(a.dataset,'train_seen',limit=a.limit_train),batch_size=a.batch_size,shuffle=True,collate_fn=collate,num_workers=2,pin_memory=True);val=DataLoader(GroundingDataset(a.dataset,'val_seen',limit=a.limit_val),batch_size=a.batch_size,shuffle=False,collate_fn=collate,num_workers=2)
 prompter=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('clip_vision_encoder')];vision=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('clip_vision_encoder')];groups=[{'params':prompter,'lr':1e-4}]+([{'params':vision,'lr':2e-5}] if vision else []);opt=torch.optim.AdamW(groups,weight_decay=.01);history=[];best=-1;started=time.time();torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,a.epochs+1):
  model.train();opt.zero_grad(set_to_none=True);tot={'localization':0.,'ranking':0.,'contrastive':0.,'total':0.};steps=0
  for step,batch in enumerate(train,1):
   images=batch['images'].to(device,dtype=torch.bfloat16);target=batch['targets'].to(device)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    out=model(images,batch['texts']);loc=soft_localization_loss(out['coarse_logits'],target);rank=ranking_loss(out['coarse_logits'],batch['candidate_masks'],batch['positive_indices']);con=contrastive(out,target);loss=(loc+.5*rank+.2*con)/a.accumulate
   loss.backward();steps+=1
   if step%a.accumulate==0 or step==len(train):torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();opt.zero_grad(set_to_none=True)
   for k,v in [('localization',loc),('ranking',rank),('contrastive',con),('total',loss*a.accumulate)]:tot[k]+=float(v.detach())
  metrics=validate(model,val,device);record={'epoch':epoch,'losses':{k:v/max(steps,1) for k,v in tot.items()},'val_seen':metrics};history.append(record);score=metrics['top1']
  if score>best:best=score;torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items() if k.startswith('prompter.') or 'lora_' in k},'mode':a.mode,'epoch':epoch,'val_seen':metrics},a.output/'best.pt')
  (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(record),flush=True)
 summary={'mode':a.mode,'official_state_entries':loaded,'trainable_parameters':trainable,'total_parameters':sum(p.numel() for p in model.parameters()),'epochs_completed':len(history),'seconds':time.time()-started,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'best_val_seen_top1':best,'history':history,'test_unseen_read':False,'text_encoder_trainable':False,'sam_used':False}
 (a.output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
