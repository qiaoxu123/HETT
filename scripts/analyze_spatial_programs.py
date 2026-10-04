#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects
from multiagent.spatial_program.binding import ambiguous_bindings,binding_coverage
from multiagent.spatial_program.parser import parse_spatial_program
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert 'test_unseen' not in str(a.output)
 objects=get_city_refer_objects(a.data_root/'cityrefer/objects.json',a.data_root/'cityrefer/processed_descriptions.json');result={}
 for split in ('train_seen','val_seen','val_unseen'):
  count=Counter();episodes=json.loads((a.data_root/'processed_citynav'/f'citynav_{split}.json').read_text());elapsed=time.perf_counter()
  for e in episodes:
   m=f"{e['area']}_block_{e['block']}";o=objects[m][int(e['object_ids'][0])];ann=int(e['ann_ids'][0]);pgr=parse_spatial_program(o.descriptions[ann],o.processed_descriptions[ann].landmarks)
   count['instructions']+=1;count[f'clauses_{min(len(pgr.segments),3)}'+('+' if len(pgr.segments)>=3 else '')]+=1;count['multi_anchor']+=sum(x.role=='reference' for x in pgr.entities)>1;count['with_program']+=bool(pgr.clauses);count['binding_ambiguous']+=ambiguous_bindings(pgr);count['bound_relations']+=binding_coverage(pgr)
   for c in pgr.clauses:count[f'relation_{c.relation}']+=1
  count['parser_ms_per_instruction']=1000*(time.perf_counter()-elapsed)/max(len(episodes),1);result[split]=dict(count)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
