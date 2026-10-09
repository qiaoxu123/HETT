#!/usr/bin/env python3
"""Read saved experiment state without touching the active GPU process."""
import argparse,json,re
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);opt=ap.parse_args()
r=opt.run.resolve();log=r.with_suffix('.log')
status={'run':str(r),'complete':(r/'COMPLETE.json').exists()}
if (r/'train_epoch01.json').exists():
 status['phase']='validation';status['train']=json.loads((r/'train_epoch01.json').read_text())
 status['train']={k:status['train'][k] for k in ['train_seconds','episodes','batches','peak_vram_allocated_bytes','losses']}
elif (r/'initial_full/epoch00/val_unseen_relations_no_stop.json').exists():status['phase']='training'
elif (r/'initial_quick/results.json').exists():status['phase']='initial_full_baseline'
else:status['phase']='initialization_or_quick_baseline'
if log.exists():
 with log.open('rb') as f:
  f.seek(max(0,log.stat().st_size-32768));tail=f.read().decode(errors='replace')
 progress=re.findall(r'Progress:.*?([0-9.]+)% (.*?\([0-9]+/[0-9]+\))',tail)
 if progress:status['latest_progress']=progress[-1]
 if 'Traceback (most recent call last)' in tail:status['error_tail']=tail[-2000:]
status['finished_evaluations']=[p.stem for p in sorted((r/'epoch01').glob('*.json'))]
print(json.dumps(status,ensure_ascii=False,indent=2))
