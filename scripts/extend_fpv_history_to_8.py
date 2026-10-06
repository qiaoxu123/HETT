#!/usr/bin/env python3
"""Add the eighth, older teacher-pose view to a completed FPV manifest."""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_fpv_landmark_memory_dataset import AirSimPerspectiveRenderer, pose_at, project_candidates, SPLITS

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--source-dataset',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--airsim-port',type=int,default=30001);p.add_argument('--image-size',type=int,default=224);a=p.parse_args()
 if 'test_unseen' in ' '.join(map(str,vars(a).values())):raise ValueError('test_unseen forbidden')
 rows=[json.loads(x) for x in (a.dataset/'manifest.jsonl').read_text().splitlines()];source={json.loads(x)['sample_id']:json.loads(x) for x in (a.source_dataset/'manifest.jsonl').read_text().splitlines()};episodes={s:json.loads((a.data_root/'processed_citynav'/f'citynav_{s}.json').read_text()) for s in SPLITS};objects=json.loads((a.data_root/'cityrefer/objects.json').read_text());renderer=AirSimPerspectiveRenderer('127.0.0.1',a.airsim_port,a.dataset,a.image_size);cache={};started=time.time();n=0
 for row in rows:
  split=row['split'];step=int(row['step'])
  if split not in SPLITS or step<13:continue
  if any(int(x.get('offset',-1))==13 for x in row['history_fpv']):continue
  epi=int(row['episode_id'].split(':',1)[1]);pose=pose_at(episodes[split][epi],step-13,0);key=(split,row['scene_id'],tuple(round(v,3) for v in pose))
  if key not in cache:
   image_key=f"{split}_{row['sample_id']}_t{step-13}_fpv_history13";path,_,depth=renderer.render(row['scene_id'],pose,0,image_key);cache[key]=(path,depth)
  path,depth=cache[key];geom,masks=project_candidates(source[row['sample_id']],objects[row['scene_id']],pose,a.image_size)
  label=a.dataset/'labels'/f"{row['sample_id']}_history13.npz";np.savez_compressed(label,**masks)
  row['history_fpv'].append({'step':step-13,'offset':13,'image':str(path),'pose':pose,'depth':depth,'candidates':geom,'label_file':str(label),'camera':'airsim_front_0','render_source':'CityNav Unreal AirSim RGB; pose offset 13; pitch=0'})
  row['history_fpv'].sort(key=lambda x:x['offset'])
  n+=1
  if n%100==0:print(f'extended rows={n}; unique frames={len(cache)}',flush=True)
 with (a.dataset/'manifest.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 stats=json.loads((a.dataset/'dataset_stats.json').read_text());stats['history_frames']=sum(len(r['history_fpv']) for r in rows);stats['history_offsets']=[0,1,3,5,7,9,11,13];stats['history_extension']={'added_rows':n,'unique_frames':len(cache),'seconds':time.time()-started};(a.dataset/'dataset_stats.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats['history_extension'],indent=2))
if __name__=='__main__':main()
