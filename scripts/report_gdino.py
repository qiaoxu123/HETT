import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/gdino_fixed30'
rows=json.loads((OUT/'predictions.json').read_text())
s=json.loads((OUT/'summary.json').read_text())
extra={}
for mode in ['category','target_phrase']:
    vis=[r for r in rows if r['status']=='in_view']
    full=[r for r in vis if r['visible_fraction']>=.9]
    extra[mode]={'mostly_in_view_n':len(full),'mostly_in_view_top1_hits':sum(r['predictions'][mode]['top1_iou']>=.5 for r in full),
                 'visible_top1_mean_iou':float(np.mean([r['predictions'][mode]['top1_iou'] for r in vis]))}
extra['identical_prompt_frames']=sum(r['predictions']['category']['prompt']==r['predictions']['target_phrase']['prompt'] for r in rows)
(OUT/'additional_checks.json').write_text(json.dumps(extra,indent=2))
fig,ax=plt.subplots(figsize=(7,4))
for i,mode in enumerate(['category','target_phrase']):
    vals=[s[mode]['top1_iou50_hits']/s['in_view']*100,s[mode]['out_of_view_frames_with_detections']/s['out_of_view']*100]
    bars=ax.bar(np.arange(2)+(i-.5)*.35,vals,.35,label=mode)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}%',ha='center')
ax.set_xticks([0,1]);ax.set_xticklabels(['Target in view: top-1 IoU >= 0.5','Target outside: any candidate returned'])
ax.set_ylim(0,105);ax.set_ylabel('Frames (%)');ax.legend();fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150)
print(json.dumps(extra,indent=2))
