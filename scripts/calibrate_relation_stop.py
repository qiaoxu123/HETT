#!/usr/bin/env python3
"""Seen-only calibration; held-out replay happens after a threshold is frozen."""
import argparse,json,csv
from pathlib import Path
import numpy as np

def replay(episodes,threshold):
    rows=[];tp=fp=0
    for e in episodes:
        path=np.asarray(e['path_xy']);goal=np.asarray(e['goal_xy']);stop=None
        for s in e['navigation_steps']:
            if s['stop_probability'] >= threshold:
                stop=s;path=path[:int(s['path_index'])+1];break
        ne=float(np.linalg.norm(path[-1]-goal));success=int(ne<=20.)
        distance=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
        shortest=float(np.linalg.norm(path[0]-goal))
        spl=success*shortest/max(shortest,distance,.01)
        if stop is not None:
            tp+=success;fp+=1-success
        rows.append(dict(episode_id=e['episode_id'],success=success,spl=spl,ne=ne,
                         path_xy=path.tolist(),stop=stop is not None))
    n=len(rows)
    return dict(threshold=float(threshold),episodes=n,sr=100*np.mean([e['success'] for e in rows]),
                spl=100*np.mean([e['spl'] for e in rows]),stop_TP=tp,stop_FP=fp,stops=tp+fp,
                precision=tp/(tp+fp) if tp+fp else None,false_stop_episode_rate=fp/n),rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);opt=ap.parse_args();r=opt.run.resolve()
    out=r/'calibrated_stop';out.mkdir(exist_ok=True)
    seen=json.loads((r/'epoch01/val_seen_relations_no_stop.json').read_text())['episodes']
    base,_=replay(seen,1.01)
    scores=np.array([s['stop_probability'] for e in seen for s in e['navigation_steps']])
    candidates=np.unique(np.r_[np.linspace(0,1,101),np.quantile(scores,np.linspace(.5,1,101)),1.01])
    rows=[replay(seen,float(t))[0] for t in candidates]
    for row in rows:
        row['accepted']=(row['precision'] is not None and row['precision']>=.95
            and row['false_stop_episode_rate']<=.005
            and row['sr']>=base['sr']-1e-10 and row['spl']>=base['spl']-1e-10)
    accepted=[a for a in rows if a['accepted']]
    chosen=max(accepted,key=lambda a:(a['sr'],a['spl'],a['threshold'])) if accepted else next(a for a in rows if a['threshold']==1.01)
    result=dict(selection_split='val_seen',threshold=chosen['threshold'],status='accepted_seen_threshold' if accepted else 'no_safe_seen_threshold_disable_stop',
                constraints=dict(stop_precision_min=.95,false_stop_episode_rate_max=.005,seen_sr_and_spl_no_decrease=True),
                seen_baseline=base,seen_chosen=chosen,seen_score_range=[float(scores.min()),float(scores.max())])
    # Save/freeze the selection before loading any Unseen information.
    (out/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'seen_threshold_calibration.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    evaluated=[]
    for split in ['val_seen','val_unseen']:
        episodes=seen if split=='val_seen' else json.loads((r/'epoch01'/f'{split}_relations_no_stop.json').read_text())['episodes']
        summary,eps=replay(episodes,chosen['threshold']);summary['split']=split;evaluated.append(summary)
        (out/f'{split}_replay.json').write_text(json.dumps(dict(summary=summary,episodes=eps),separators=(',',':')))
    (out/'results.json').write_text(json.dumps(evaluated,indent=2)+'\n')
    print(json.dumps(dict(selection=result,results=evaluated),indent=2))
if __name__=='__main__':main()
