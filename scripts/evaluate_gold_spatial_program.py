#!/usr/bin/env python3
"""Guarded entry point: only genuinely reviewed programs may be called Gold."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--gold',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();data=json.loads(a.gold.read_text());reviewed=[r for r in data['records'] if r.get('human_reviewed')]
 result={'status':'blocked_without_human_review','reviewed_records':len(reviewed),'required_minimum':150,'gt_target_used':False}
 if len(reviewed)<150:result['reason']='P7 requires at least 150 independently reviewed programs; auto-parser output is not a valid Gold upper bound.'
 else:result.update(status='ready_for_executor_evaluation',reason='Run P7 with reviewed program JSON; no target coordinate is admitted.')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
