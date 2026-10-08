#!/usr/bin/env python3
"""Read completed training/evaluation results; never performs training."""
import argparse,csv,json
from pathlib import Path

def summarize(out):
 if not (out/'COMPLETE.json').exists():raise RuntimeError('pipeline is not complete')
 selection=json.loads((out/'selection.json').read_text());cfg=json.loads((out/'manifest.json').read_text())['config']
 result_rows=[]
 for epoch in range(1,cfg['epochs']+1):
  for row in json.loads((out/f'eval_epoch{epoch}/results.json').read_text()):result_rows.append({'scope':'subset256','epoch':epoch,**row})
 recommended=False
 if selection['proxy_improved']:
  rows=json.loads((out/'full_validation/results.json').read_text());result_rows.extend({'scope':'full','epoch':selection['selected_epoch'],**row} for row in rows)
  ub=next(r for r in rows if r['split']=='val_unseen' and r['variant']=='baseline');ur=next(r for r in rows if r['split']=='val_unseen' and r['variant']!='baseline')
  sb=next(r for r in rows if r['split']=='val_seen' and r['variant']=='baseline');sr=next(r for r in rows if r['split']=='val_seen' and r['variant']!='baseline')
  recommended=ur['sr']>ub['sr'] and ur['spl']>=ub['spl'] and sr['sr']>=sb['sr']
 final={'epochs_trained':cfg['epochs'],'selected_epoch':selection['selected_epoch'],'proxy_improved':selection['proxy_improved'],'recommended_over_waypoint_baseline':recommended,'runs':result_rows}
 (out/'results.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
 fields=['scope','epoch','split','variant','episodes','sr','oracle_sr','spl','ne','delta_sr_pp']
 with (out/'results.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for row in result_rows:writer.writerow({k:row[k] for k in fields})
 return final
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();print(json.dumps(summarize(args.output)))
