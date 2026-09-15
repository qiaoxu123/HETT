import sys,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/paired_screen_s0';summary={}
for split in ['val_seen','val_unseen']:
    predictions={m:torch.load(RUN/m/'evaluation'/f'{split}_predictions.pt',map_location='cpu',weights_only=False) for m in ['human','straight']}
    assert set(predictions['human'])==set(predictions['straight'])
    rows=[];clusters=defaultdict(list)
    for key in predictions['human']:
        ne={m:float(np.linalg.norm(np.array(p[key]['trajectory'][-1])[:2]-np.array(p[key]['goal'])[:2])) for m,p in predictions.items()}
        row={'id':list(key),'human_ne':ne['human'],'straight_ne':ne['straight'],'human_success':ne['human']<=20,'straight_success':ne['straight']<=20}
        rows.append(row);clusters[key[:2]].append(int(row['straight_success'])-int(row['human_success']))
    vals=np.array([sum(v) for v in clusters.values()]);ns=np.array([len(v) for v in clusters.values()])
    rng=np.random.default_rng(0);idx=rng.integers(0,len(vals),(5000,len(vals)))
    boot=100*vals[idx].sum(axis=1)/ns[idx].sum(axis=1)
    metrics={m:json.loads((RUN/m/'evaluation'/f'{split}_metrics.json').read_text()) for m in ['human','straight']}
    summary[split]={'episodes':len(rows),'metrics':metrics,'sr_gain_pp':metrics['straight']['sr']-metrics['human']['sr'],
                   'paired_object_bootstrap95_pp':np.quantile(boot,[.025,.975]).tolist(),
                   'new_successes':sum(r['straight_success'] and not r['human_success'] for r in rows),
                   'lost_successes':sum(r['human_success'] and not r['straight_success'] for r in rows),'rows':rows}
(RUN/'comparison.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(1,2,figsize=(9,4))
for ax,metric,label in zip(axs,['sr','ne'],['Success rate (%)','Final distance (m)']):
    for i,mode in enumerate(['human','straight']):
        values=[summary[s]['metrics'][mode][metric] for s in summary]
        bars=ax.bar(np.arange(2)+(i-.5)*.36,values,.36,label=mode)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v,f'{v:.2f}',ha='center',va='bottom')
    ax.set_xticks([0,1]);ax.set_xticklabels(['Seen','Unseen']);ax.set_ylabel(label);ax.legend();ax.margins(y=.2)
fig.tight_layout();fig.savefig(RUN/'comparison.png',dpi=160)
print(json.dumps({s:{k:v for k,v in r.items() if k not in ['rows','metrics']} for s,r in summary.items()},indent=2))
