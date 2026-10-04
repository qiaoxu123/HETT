#!/usr/bin/env python3
"""Build a stratified *review queue* for spatial-program annotation.

No automatically produced record is represented as human gold.  Reviewers can
edit ``program`` and set ``human_reviewed=true``; parser metrics ignore the
unreviewed records unless --allow-silver is explicitly requested.
"""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects
from multiagent.spatial_program.parser import parse_spatial_program

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'analysis/spatial_program_gold.json');p.add_argument('--count',type=int,default=200);a=p.parse_args()
 episodes=json.loads((a.data_root/'processed_citynav/citynav_val_seen.json').read_text());objects=get_city_refer_objects(a.data_root/'cityrefer/objects.json',a.data_root/'cityrefer/processed_descriptions.json');buckets=defaultdict(list)
 for i,e in enumerate(episodes):
  m=f"{e['area']}_block_{e['block']}";o=objects[m][int(e['object_ids'][0])];ann=int(e['ann_ids'][0]);instruction=o.descriptions[ann];refs=o.processed_descriptions[ann].landmarks
  program=parse_spatial_program(instruction,refs);rels=tuple(sorted({c.relation for c in program.clauses}));key='multi_clause' if len(program.segments)>1 else rels[0] if rels else 'no_relation'
  buckets[key].append({'episode_index':i,'map_name':m,'instruction':instruction,'program':program.to_dict(),'annotation_source':'semi_automatic_review_candidate','human_reviewed':False,'review_notes':''})
 keys=sorted(buckets);out=[];cursor={k:0 for k in keys}
 while len(out)<min(a.count,sum(map(len,buckets.values()))):
  changed=False
  for k in keys:
   if cursor[k]<len(buckets[k]) and len(out)<a.count:out.append(buckets[k][cursor[k]]);cursor[k]+=1;changed=True
  if not changed:break
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'schema_version':1,'split':'val_seen','records':out,'human_reviewed_count':0,'warning':'UNREVIEWED SILVER QUEUE; not human gold'},indent=2)+'\n');print(json.dumps({'records':len(out),'buckets':{k:min(cursor[k],len(v)) for k,v in buckets.items()}},indent=2))
if __name__=='__main__':main()
