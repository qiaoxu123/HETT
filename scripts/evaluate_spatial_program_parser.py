#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from multiagent.spatial_program.metrics import parser_metrics
from multiagent.spatial_program.parser import parse_spatial_program
from multiagent.spatial_program.schema import SpatialProgram
def main():
 p=argparse.ArgumentParser();p.add_argument('--gold',type=Path,default=ROOT/'analysis/spatial_program_gold.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-silver',action='store_true');a=p.parse_args();data=json.loads(a.gold.read_text());records=[r for r in data['records'] if r.get('human_reviewed')]
 if not records and not a.allow_silver:raise SystemExit('No human-reviewed records. Review the queue or pass --allow-silver and report silver agreement, not gold accuracy.')
 records=records or data['records'];pairs=[(parse_spatial_program(r['instruction']),SpatialProgram.from_dict(r['program'])) for r in records];m=parser_metrics(pairs);m.update({'evaluation_kind':'human_gold' if records and records[0].get('human_reviewed') else 'silver_self_agreement','warning':None if records and records[0].get('human_reviewed') else 'Not a human gold parser evaluation'})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
if __name__=='__main__':main()
