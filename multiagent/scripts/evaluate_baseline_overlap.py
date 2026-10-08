import json,numpy as np
from multiagent.visual_goal.diagnosis_data import OUT,load_split,write_json
from multiagent.visual_goal.failure_analysis import summarize
from multiagent.visual_goal.overlap import BINS,bin_mask
out={}
for split in ('val_seen','val_unseen'):
 q,c=load_split(split);cache=OUT/'cache'
 if not (cache/f'global_{split}_q.npy').exists():continue
 s=np.load(cache/f'global_{split}_q.npy')@np.load(cache/f'global_{split}_c.npy').T
 ov=np.load(cache/f'overlap_{split}.npz');pi=[next(j for j,x in enumerate(c) if x['scene_key']==r['scene_key']) for r in q];ix=np.arange(len(q))
 result={'all':summarize(s,q,c)}
 for k in ('query_coverage','template_coverage','iou'):
  p=ov[k][ix,pi];result[k]={label:summarize(s,q,c,bin_mask(p,lo,hi)) for lo,hi,label in BINS}
 out[split]=result
 print(split,result['all'],flush=True)
 for k in ('query_coverage','template_coverage'):
  print(k,[(a,v['n'],v.get('R@1')) for a,v in result[k].items()],flush=True)
write_json(OUT/'baseline_overlap.json',out)
