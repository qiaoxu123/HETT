#!/usr/bin/env python3
"""Attach projected reference-mask labels to existing real-pose RGB."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects
from multiagent.rsrefseg2_grounding.dataset import build_frame_labels,leakage_audit

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-dataset',type=Path,required=True);p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,required=True);p.add_argument('--image-size',type=int,default=256);p.add_argument('--object-disjoint-seen',action='store_true',help='Diagnostic only: CityNav seen splits share anchors and this removes all current training records');a=p.parse_args()
 if 'test_unseen' in ' '.join(map(str,vars(a).values())):raise ValueError('test_unseen forbidden')
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'labels').mkdir(exist_ok=True);objects=get_city_refer_objects(a.data_root/'cityrefer/objects.json',a.data_root/'cityrefer/processed_descriptions.json');episodes={s:json.loads((a.data_root/'processed_citynav'/f'citynav_{s}.json').read_text()) for s in ('train_seen','val_seen','val_unseen')};source=[json.loads(x) for x in (a.source_dataset/'manifest.jsonl').read_text().splitlines()];records=[];reject=Counter()
 for row in source:
  split,index=row['episode_id'].split(':');label=build_frame_labels(row,episodes[split][int(index)],objects[row['map_name']],a.image_size)
  if label is None:reject[f'{split}:referenced_not_visible']+=1;continue
  identifier=row['sample_id'];np.savez_compressed(a.output/'labels'/f'{identifier}.npz',gt_mask=label.pop('gt_mask'),coarse_target=label.pop('coarse_target'),**{f"candidate_{k}":v for k,v in label.pop('candidate_masks').items()})
  record={**row,**label,'image':str((a.source_dataset/row['image']).resolve()),'label_file':str((a.output/'labels'/f'{identifier}.npz').resolve()),'candidate_landmarks':label['candidates'],'same_scene_hard_negative_ids':[c['landmark_id'] for c in label['candidates'] if c['landmark_id'] not in label['referenced_landmark_ids']]};records.append(record)
 # Optional stricter diagnostic. Official CityNav seen splits deliberately
 # share maps/anchors; enabling this on the current data removes every training
 # record, so the formal protocol preserves that documented seen-split design.
 held_out={f"{r['map_name']}:{oid}" for r in records if r['split']!='train_seen' for oid in r['referenced_landmark_ids']}
 before=len(records)
 if a.object_disjoint_seen:records=[r for r in records if r['split']!='train_seen' or not any(f"{r['map_name']}:{oid}" in held_out for oid in r['referenced_landmark_ids'])]
 removed=before-len(records)
 with (a.output/'manifest.jsonl').open('w') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 stats={'samples':len(records),'by_split':dict(Counter(r['split'] for r in records)),'train_records_removed_for_object_disjointness':removed,'observable_fraction_by_split':{s:sum(r['split']==s for r in records)/max(sum(x['split']==s for x in source),1) for s in ('train_seen','val_seen','val_unseen')},'candidate_density':float(np.mean([len(r['candidates']) for r in records])) if records else 0,'same_scene_hard_negative_fraction':float(np.mean([bool(r['same_scene_hard_negative_ids']) for r in records])) if records else 0,'rejections':dict(reject),'leakage':leakage_audit(records),'source_dataset':str(a.source_dataset.resolve()),'test_unseen_read':False}
 (a.output/'dataset_stats.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
