#!/usr/bin/env python3
"""Split/view visibility-proxy, trajectory and rendered-image sanity audit."""
from __future__ import annotations
import argparse,json,random
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--image-sample-per-view',type=int,default=300);a=p.parse_args()
 rows=[json.loads(x) for x in (a.dataset/'manifest.jsonl').read_text().splitlines()];summary={'rows':len(rows),'by_split':dict(Counter(r['split'] for r in rows)),'split_view':{},'rendered_image_quality_sample':{},'limitations':{'visibility':'projected CityRefer geometry only; occlusion unavailable','depth':'AirSim smoke produced no usable depth pixels'}}
 for split in ('train_seen','val_seen','val_unseen'):
  subset=[r for r in rows if r['split']==split]; view_stats={}
  for view in ('fpv','oblique30','oblique45'):
   visible=[];areas=[];ratios=[];distances=[];candidates=[]
   for row in subset:
    obs=row['view_observations'][view];ref=set(map(int,row['referenced_landmark_ids']));positive=[c for c in obs['candidates'] if int(c['landmark_id']) in ref]
    visible.append(any(c['visible'] for c in positive));areas.extend(c['pixel_area'] for c in positive);ratios.extend(c['visible_ratio'] for c in positive);distances.extend(c['distance_m'] for c in positive);candidates.extend(obs['candidates'])
   current=[];prior=[];never=[]
   if view=='fpv':
    for row in subset:
     current.append(bool(row['target_landmark_visible_now']));prior.append(bool(not row['target_landmark_visible_now'] and row['target_landmark_seen_before']));never.append(bool(not row['target_landmark_visible_now'] and not row['target_landmark_seen_before']))
   view_stats[view]={'observations':len(subset),'projected_referenced_visible_n':int(sum(visible)),'projected_referenced_visible_rate':float(np.mean(visible)) if visible else None,'projected_pixel_area_mean':float(np.mean(areas)) if areas else None,'projected_visible_ratio_mean':float(np.mean(ratios)) if ratios else None,'reference_distance_mean_m':float(np.mean(distances)) if distances else None,'candidate_outside_projected_fov_fraction':float(np.mean([not c['visible'] for c in candidates])) if candidates else None}
   if view=='fpv':view_stats[view].update({'current_visible':int(sum(current)),'current_invisible_previously_seen':int(sum(prior)),'never_seen_in_8_view_window':int(sum(never))})
  summary['split_view'][split]=view_stats
  for view in ('fpv','oblique30','oblique45'):
   paths=[r['view_observations'][view]['image'] for r in subset]
   rng=random.Random(0);paths=rng.sample(paths,min(len(paths),a.image_sample_per_view));sat=[];luma=[];std=[]
   for path in paths:
    image=cv2.imread(path)
    if image is None:continue
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);sat.append(float(hsv[:,:,1].mean()/255));luma.append(float(gray.mean()/255));std.append(float(gray.std()/255))
   summary['rendered_image_quality_sample'][f'{split}:{view}']={'n':len(std),'mean_saturation_0_1':float(np.mean(sat)) if sat else None,'mean_luminance_0_1':float(np.mean(luma)) if luma else None,'mean_luminance_std_0_1':float(np.mean(std)) if std else None}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
