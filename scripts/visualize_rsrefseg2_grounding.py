#!/usr/bin/env python3
"""Export qualitative CityNav grounding panels for frozen/adapted RSRefSeg2."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.rsrefseg2_grounding.io import GroundingDataset,collate
from multiagent.rsrefseg2_grounding.metrics import candidate_scores
from multiagent.rsrefseg2_grounding.official_adapter import OfficialCoarseGrounder,configure_trainable

def load(root,weights,adapted=None):
 model=OfficialCoarseGrounder(root);model.load_official_checkpoint(weights)
 if adapted:model.load_state_dict(torch.load(adapted,map_location='cpu',weights_only=False)['model'],strict=False)
 configure_trainable(model,'frozen');return model.cuda().to(torch.bfloat16).eval()

def infer(model,loader):
 output=[]
 with torch.inference_mode():
  for batch in loader:
   with torch.autocast('cuda',dtype=torch.bfloat16):value=model(batch['images'].cuda().to(torch.bfloat16),batch['texts'])['coarse_logits'].float().cpu()
   output.extend(value[:,0])
 return output

def overlay(rgb,heat,color_map=cv2.COLORMAP_TURBO):
 h=cv2.resize(heat.numpy(),(rgb.shape[1],rgb.shape[0]));h=(h-h.min())/(np.ptp(h)+1e-8);colour=cv2.applyColorMap(np.uint8(255*h),color_map)
 return cv2.addWeighted(rgb,.56,colour,.44,0)

def panel(row,payload,base,frozen,adapted,group):
 rgb=cv2.imread(row['image']);gt=np.uint8(payload['gt_mask'])*255
 views=[rgb,overlay(rgb,base),overlay(rgb,frozen),overlay(rgb,adapted)];names=['RGB + GT contour','SigLIP2 dense','RSRefSeg2 frozen','RSRefSeg2 adapted']
 for view,name in zip(views,names):
  contour,_=cv2.findContours(gt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(view,contour,-1,(0,255,0),2);cv2.putText(view,name,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),2,cv2.LINE_AA)
 scores=candidate_scores(adapted,[payload[f"candidate_{x['landmark_id']}"] for x in row['candidates']]);order=np.argsort(-scores);pred=row['candidates'][int(order[0])];positive=set(row['referenced_landmark_ids']);pos=max(scores[i] for i,c in enumerate(row['candidates']) if c['landmark_id'] in positive);neg=max([scores[i] for i,c in enumerate(row['candidates']) if c['landmark_id'] not in positive] or [float('nan')])
 canvas=np.concatenate(views,axis=1);footer=np.zeros((120,canvas.shape[1],3),np.uint8);lines=[f"group={group} sample={row['sample_id']} step={row['step']}",row['instruction'][:190],f"GT={row['referenced_landmark_names']} predicted={pred['name']} positive={pos:.3f} hard-negative={neg:.3f} margin={pos-neg:.3f}"]
 for i,line in enumerate(lines):cv2.putText(footer,line,(8,25+32*i),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1,cv2.LINE_AA)
 return np.concatenate([canvas,footer],axis=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--official-root',type=Path,required=True);p.add_argument('--official-checkpoint',type=Path,required=True);p.add_argument('--adapted-checkpoint',type=Path,required=True);p.add_argument('--frozen-records',type=Path,required=True);p.add_argument('--adapted-records',type=Path,required=True);p.add_argument('--count',type=int,default=20);a=p.parse_args()
 if 'test_unseen' in ' '.join(map(str,vars(a).values())):raise ValueError('test_unseen forbidden')
 frozen_rec=torch.load(a.frozen_records,map_location='cpu',weights_only=False)['val_unseen']['full'];adapted_rec=torch.load(a.adapted_records,map_location='cpu',weights_only=False)['val_unseen']['full'];groups={'success':[i for i,x in enumerate(adapted_rec) if x['top1']],'failure':[i for i,x in enumerate(adapted_rec) if not x['top1']],'frozen_to_adapted':[i for i,(x,y) in enumerate(zip(frozen_rec,adapted_rec)) if not x['top1'] and y['top1']],'adapted_still_failed':[i for i,(x,y) in enumerate(zip(frozen_rec,adapted_rec)) if not x['top1'] and not y['top1']]}
 chosen=[]
 for name,indices in groups.items():
  take=indices[:a.count];chosen.extend((name,x) for x in take)
 ds=GroundingDataset(a.dataset,'val_unseen');indices=[x for _,x in chosen];loader=DataLoader(Subset(ds,indices),batch_size=4,collate_fn=collate)
 frozen=load(a.official_root,a.official_checkpoint);frozen_heat=infer(frozen,loader);del frozen;torch.cuda.empty_cache()
 adapted=load(a.official_root,a.official_checkpoint,a.adapted_checkpoint);adapted_heat=infer(adapted,loader);del adapted;torch.cuda.empty_cache()
 # Baseline dense heatmaps use the same text/image only; import avoids duplicate implementation.
 from evaluate_rsrefseg2_grounding import SiglipBaseline
 baseline=SiglipBaseline('google/siglip2-base-patch16-256');base_heat=[]
 for index in indices:
  item=ds[index];base_heat.append(baseline.row(item['row'],item['payload'],'dense'))
 manifest=[];a.output.mkdir(parents=True,exist_ok=True)
 for n,((group,index),b,f,z) in enumerate(zip(chosen,base_heat,frozen_heat,adapted_heat)):
  item=ds[index];folder=a.output/group;folder.mkdir(exist_ok=True);path=folder/f"{n:03d}_{item['row']['sample_id']}.png";cv2.imwrite(str(path),panel(item['row'],item['payload'],b,f,z,group));manifest.append({'group':group,'sample_id':item['row']['sample_id'],'path':str(path)})
 (a.output/'manifest.json').write_text(json.dumps({'samples':manifest,'test_unseen_read':False},indent=2)+'\n')
if __name__=='__main__':main()
