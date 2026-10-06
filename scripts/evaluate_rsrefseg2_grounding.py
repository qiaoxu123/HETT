#!/usr/bin/env python3
"""Evaluate comparable candidate/localization metrics without navigation."""
from __future__ import annotations
import argparse,csv,json,sys,time
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.rsrefseg2_grounding.io import GroundingDataset,collate,language
from multiagent.rsrefseg2_grounding.metrics import aggregate,frame_result,retrieval_result
from multiagent.rsrefseg2_grounding.official_adapter import OfficialCoarseGrounder,configure_trainable

def load_grounder(a,checkpoint=None):
 m=OfficialCoarseGrounder(a.official_root);m.load_official_checkpoint(a.official_checkpoint)
 if checkpoint:
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False);m.load_state_dict(saved['model'],strict=False)
 configure_trainable(m,'frozen');return m.to('cuda',dtype=torch.bfloat16).eval()
def evaluate_grounder(model,dataset,batch_size,language_mode,gt_crop=False):
 loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate,num_workers=2);results=[]
 with torch.inference_mode():
  for batch in loader:
   texts=[language(r,language_mode) for r in batch['rows']]
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch['images'].cuda().to(torch.bfloat16),texts)
   for i,(row,payload) in enumerate(zip(batch['rows'],batch['payloads'])):
    if not gt_crop:results.append(frame_result(row,out['coarse_logits'][i].float().cpu(),payload));continue
    # Diagnostic only: candidate GT boxes generate crops, but are not used by full-image inference.
    image=batch['images'][i];crops=[]
    for c in row['candidates']:
     x0,y0,x1,y1=c['bbox_xyxy'];crop=F.interpolate(image[:,y0:y1,x0:x1][None],size=(256,256),mode='bilinear',align_corners=False)
     crops.append(crop[0])
    # Candidate density has a long tail.  Chunking keeps this diagnostic below
    # the memory footprint of normal full-frame training and changes no score.
    coarse=[]
    for start in range(0,len(crops),max(1,batch_size)):
     chunk=torch.stack(crops[start:start+max(1,batch_size)]).cuda().to(torch.bfloat16)
     with torch.autocast('cuda',dtype=torch.bfloat16):
      coarse.append(model(chunk,[texts[i]]*len(chunk))['coarse_logits'].float())
    coarse=torch.cat(coarse)
    scores=[float(x.flatten().topk(max(1,x.numel()//20)).values.mean()) for x in coarse]
    results.append(retrieval_result(row,scores))
 return results

class SiglipBaseline:
 def __init__(self,name):
  from transformers import AutoModel,AutoProcessor
  self.processor=AutoProcessor.from_pretrained(name);self.model=AutoModel.from_pretrained(name).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
 @torch.inference_mode()
 def row(self,row,payload,mode):
  image=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB);text=row['instruction'];tok=self.processor(text=[text],padding=True,truncation=True,max_length=64,return_tensors='pt');tok={k:v.cuda() for k,v in tok.items() if k in ('input_ids','attention_mask')};t=self.model.text_model(**tok).pooler_output.float()
  if mode=='dense':
   px=self.processor(images=[image],return_tensors='pt')['pixel_values'].cuda().to(torch.bfloat16);v=self.model.vision_model(pixel_values=px).last_hidden_state.float();sim=F.normalize(v,dim=-1)@F.normalize(t,dim=-1).T;side=int(np.sqrt(v.shape[1]));return F.interpolate(sim.squeeze(-1).reshape(1,1,side,side),size=(256,256),mode='bilinear',align_corners=False)[0,0].cpu()
  crops=[]
  for c in row['candidates']:
   x0,y0,x1,y1=c['bbox_xyxy'];crop=image[y0:y1,x0:x1] if x1>x0 and y1>y0 else image
   # Very distant landmarks can occupy only one or two pixels.  PIL records the
   # channel layout unambiguously, but older HF processors still infer a 1-pixel
   # side as the channel dimension; enlarge before preprocessing.
   pil=Image.fromarray(crop)
   if min(pil.size)<4:pil=pil.resize((max(4,pil.width),max(4,pil.height)),Image.Resampling.BILINEAR)
   crops.append(pil)
  px=self.processor(images=crops,return_tensors='pt')['pixel_values'].cuda().to(torch.bfloat16);features=self.model.get_image_features(pixel_values=px).float();textf=self.model.get_text_features(**tok).float();scores=(F.normalize(features,dim=-1)@F.normalize(textf,dim=-1).T).squeeze(1).cpu();heat=torch.full((256,256),float(scores.min()-1))
  for c,s in zip(row['candidates'],scores):heat[torch.as_tensor(payload[f"candidate_{c['landmark_id']}"]).bool()]=float(s)
  return heat

def grouped(rows,records):
 groups=defaultdict(list)
 for metric,row in zip(rows,records):
  pos=row['positives'][0];category=pos['category'];area=pos['visible_pixels'];density=len(row['candidates']);distance=row['goal_distance_m'];groups[f'category:{category}'].append(metric);groups['size:small' if area<256 else 'size:medium' if area<2048 else 'size:large'].append(metric);groups['density:1-4' if density<=4 else 'density:5-8' if density<=8 else 'density:9+'].append(metric);groups['distance:'+row['distance_bucket']].append(metric)
 return {k:aggregate(v) for k,v in groups.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--official-root',type=Path,required=True);p.add_argument('--official-checkpoint',type=Path,required=True);p.add_argument('--adapted-checkpoint',type=Path);p.add_argument('--method',choices=('rsref','siglip','oracle'),required=True);p.add_argument('--model-name',default='google/siglip2-base-patch16-256');p.add_argument('--batch-size',type=int,default=2);p.add_argument('--limit',type=int,default=0);p.add_argument('--gt-crop',action='store_true');p.add_argument('--language-ablation',action='store_true');a=p.parse_args()
 if 'test_unseen' in ' '.join(map(str,vars(a).values())):raise ValueError('test_unseen forbidden')
 a.output.mkdir(parents=True,exist_ok=True);started=time.time();torch.cuda.reset_peak_memory_stats();all_metrics={};per_records={}
 if a.method=='rsref':model=load_grounder(a,a.adapted_checkpoint)
 elif a.method=='siglip':model=SiglipBaseline(a.model_name)
 for split in ('val_seen','val_unseen'):
  ds=GroundingDataset(a.dataset,split,limit=a.limit);records=ds.rows
  if a.method=='rsref':
   variants=['full','landmark_only','name_category','no_spatial'] if a.language_ablation else ['full'];outputs={v:evaluate_grounder(model,ds,a.batch_size,v,a.gt_crop) for v in variants}
  elif a.method=='siglip':
   outputs={mode:[] for mode in ('dense','region')}
   for item in ds:
    for mode in outputs:outputs[mode].append(frame_result(item['row'],model.row(item['row'],item['payload'],mode),item['payload']))
  else:
   outputs={'gt_reference_mask':[]}
   for item in ds:outputs['gt_reference_mask'].append(frame_result(item['row'],torch.from_numpy(item['payload']['gt_mask']).float()*20,item['payload']))
  all_metrics[split]={name:aggregate(values) for name,values in outputs.items()};all_metrics[split]['groups']=grouped(next(iter(outputs.values())),records);per_records[split]={name:values for name,values in outputs.items()}
 result={'method':a.method,'adapted_checkpoint':str(a.adapted_checkpoint) if a.adapted_checkpoint else None,'gt_crop':a.gt_crop,'splits':all_metrics,'seconds':time.time()-started,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,'test_unseen_read':False,'referenced_mask_model_input':False}
 (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(per_records,a.output/'per_record.pt');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
