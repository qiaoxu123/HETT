#!/usr/bin/env python3
"""Generate deterministic top-down rule-debug examples."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects
from multiagent.geometry_reasoning.anchors import resolve_anchors
from multiagent.geometry_reasoning.debug_viz import render
from multiagent.geometry_reasoning.parser import parse_geometry
from multiagent.geometry_reasoning.scorer import GeometryReasoner
from multiagent.mapdata import MAP_BOUNDS
from multiagent.navigation_state import nms_topk_from_belief
from multiagent.scene_grounding.dataset import trajectory_pose
def world(m,r,c):b=MAP_BOUNDS[m];return b.x_min+(c+.5)*410/30,b.y_max-(r+.5)*410/30
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--b0-cache-root',type=Path,required=True);p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=100);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 objects=get_city_refer_objects(a.data_root/'cityrefer/objects.json',a.data_root/'cityrefer/processed_descriptions.json');weight=json.loads(a.metrics.read_text())['variants']['G7_b0_geometry_start']['lambda'];reasoner=GeometryReasoner()
 for split in ('val_seen','val_unseen'):
  out=a.output/split;out.mkdir(exist_ok=True);episodes=json.loads((a.data_root/'processed_citynav'/f'citynav_{split}.json').read_text());beliefs=torch.load(a.b0_cache_root/f'{split}_b0_eall.pt',map_location='cpu',weights_only=True)
  generated=0
  for idx,(e,b) in enumerate(zip(episodes,beliefs)):
   if generated>=a.count:break
   m=f"{e['area']}_block_{e['block']}";obj=objects[m][int(e['object_ids'][0])];ann=int(e['ann_ids'][0]);program=parse_geometry(obj.descriptions[ann],obj.processed_descriptions[ann].landmarks);anchors=resolve_anchors(program,objects[m].values());resolved=[x for g in anchors for x in g['matches']]
   if not resolved:continue
   peaks=nms_topk_from_belief(b,top_k=16,kernel_size=3);xy=[world(m,x.row,x.col) for x in peaks];pose=trajectory_pose(e['trajectory'][0]);detail=reasoner.score(program,anchors,xy,pose[:2],pose.yaw,None,True);final=np.log(np.maximum([x.probability for x in peaks],1e-12))+weight*np.asarray([d['total'] for d in detail]);image=render(xy,resolved[0]['centroid'],pose[:2],e['target_positions'][-1][:2],final.tolist());cv2.imwrite(str(out/f'{generated:04d}.png'),image);(out/f'{generated:04d}.json').write_text(json.dumps({'instruction':obj.descriptions[ann],'program':program.to_dict(),'anchors':anchors,'candidates':[{'xy':p,'b0':peaks[i].probability,'geometry':detail[i],'final':float(final[i])} for i,p in enumerate(xy)]},indent=2)+'\n');generated+=1
if __name__=='__main__':main()
