#!/usr/bin/env python3
"""Frozen bounded-fusion diagnostic; zero optimizer steps."""
import csv,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/bounded_sbf_fusion'
OUT.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,'-u',str(ROOT/'scripts/diagnose_candidate_controller.py'),
     '--config',str(ROOT/'configs/diagnostics/bounded_sbf_fusion.json'),'--output',str(OUT)]
with (OUT/'run.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
rows=json.loads((OUT/'results.json').read_text())
with (OUT/'results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
parity=[]
for split in ('val_seen','val_unseen'):
    ref=ROOT.parent/'hett-frozen-sbf-selector/artifacts/frozen_sbf_selector'/f'{split}_baseline.json'
    a=json.loads(ref.read_text())['episodes'];b=json.loads((OUT/f'{split}_baseline.json').read_text())['episodes']
    assert [x['episode_id'] for x in a]==[x['episode_id'] for x in b]
    assert all(x['path_xy']==y['path_xy'] for x,y in zip(a,b))
    parity.append(split)
(OUT/'BASELINE_PARITY.json').write_text(json.dumps({'verified_splits':parity},indent=2)+'\n')
print(json.dumps(rows,indent=2),flush=True)
