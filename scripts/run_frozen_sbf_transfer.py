#!/usr/bin/env python3
"""Run paired, frozen HETT/SBF selector transfer on the fixed 256-episode subsets."""
import csv,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/frozen_sbf_selector'
SMOKE=ROOT/'artifacts/frozen_sbf_smoke'
def main():
    smoke_pid=int(sys.argv[1])
    while not (SMOKE/'COMPLETE.json').exists():
        os.kill(smoke_pid,0)
        time.sleep(10)
    smoke=json.loads((SMOKE/'results.json').read_text())
    assert len(smoke)==4 and all(r['episodes']==8 for r in smoke)
    OUT.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(ROOT/'scripts/diagnose_candidate_controller.py'),
         '--config',str(ROOT/'configs/diagnostics/frozen_sbf_selector.json'),'--output',str(OUT)]
    with (OUT/'run.log').open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    rows=json.loads((OUT/'results.json').read_text())
    fields=sorted({k for r in rows for k in r})
    with (OUT/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    # Independently completed baseline must reproduce on the same episode IDs.
    parity=[]
    for split in ('val_seen','val_unseen'):
        original=ROOT.parent/'hett-heatmap-diagnosis/artifacts/candidate_controller'/f'{split}_waypoint_api.json'
        if original.exists():
            a=json.loads(original.read_text())['episodes']
            b=json.loads((OUT/f'{split}_baseline.json').read_text())['episodes']
            assert [e['episode_id'] for e in a]==[e['episode_id'] for e in b]
            assert all(x['path_xy']==y['path_xy'] for x,y in zip(a,b))
            parity.append(split)
    (OUT/'BASELINE_PARITY.json').write_text(json.dumps({'verified_splits':parity},indent=2))
    print(json.dumps(rows,indent=2),flush=True)
if __name__=='__main__':main()
