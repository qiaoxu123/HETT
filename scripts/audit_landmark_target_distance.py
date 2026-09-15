"""Audit supplied landmark geometry against evaluation target; never policy input."""
import sys,json
from pathlib import Path
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.dataset.generate import generate_episodes_from_mturk_trajectories
from multiagent.maps.landmark_map import LandmarkMap

OUT=ROOT/'runs/landmark_target_distance'
OUT.mkdir(parents=True,exist_ok=True)
objects=get_city_refer_objects()
rows=[]; summaries={}
for split in ['val_seen','val_unseen']:
    episodes=generate_episodes_from_mturk_trajectories(objects,load_mturk_trajectories(split,'all',50))
    for ep in episodes:
        names=ep.description_landmarks
        lms=LandmarkMap._search_landmarks_by_name(ep.map_name,names) if names else []
        target=Point(ep.target_position.x,ep.target_position.y)
        polygons=[lm.contour_polygon if lm.contour_polygon.is_valid else lm.contour_polygon.buffer(0) for lm in lms]
        polygons=[p for p in polygons if not p.is_empty]
        row={'split':split,'map':ep.map_name,'object_id':ep.target_object.id,'description_id':ep.description_id,
             'description':ep.target_description,'target_xy':list(ep.target_position.xy),'queries':names,
             'matched_names':[lm.name for lm in lms], 'exact_names':all(q==lm.name for q,lm in zip(names,lms)),
             'contours':[list(map(list,lm.contour)) for lm in lms]}
        if polygons:
            union=unary_union(polygons)
            row.update(distance_m=float(target.distance(union)),inside=bool(union.covers(target)),
                       nearest_centroid_distance_m=float(min(target.distance(p.centroid) for p in polygons)))
        rows.append(row)
    part=[r for r in rows if r['split']==split]; usable=[r for r in part if 'distance_m' in r]
    d=np.array([r['distance_m'] for r in usable]); c=np.array([r['nearest_centroid_distance_m'] for r in usable])
    summaries[split]={'episodes':len(part),'with_contour':len(usable),'without_contour':len(part)-len(usable),
       'inside_percent':100*np.mean([r['inside'] for r in usable]),
       'distance_mean_m':float(d.mean()),'distance_median_m':float(np.median(d)),
       'distance_p90_m':float(np.quantile(d,.9)),'centroid_distance_median_m':float(np.median(c)),
       'within_percent':{str(t):float(100*np.mean(d<=t)) for t in [5,10,20,50]},
       'coverage_all_episodes_percent':{str(t):float(100*np.sum(d<=t)/len(part)) for t in [0,5,10,20,50]},
       'nonexact_name_episodes':sum(not r['exact_names'] for r in usable)}
(OUT/'results.json').write_text(json.dumps({'summary':summaries,'rows':rows},indent=2,ensure_ascii=False))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for split in summaries:
    d=np.sort([r['distance_m'] for r in rows if r['split']==split and 'distance_m' in r])
    axes[0].plot(d,np.arange(1,len(d)+1)/len(d)*100,label=split)
axes[0].set(xlabel='Distance to nearest supplied landmark region (m)',ylabel='Cumulative coverage (%)',xlim=(0,150),ylim=(0,100))
axes[0].legend(); axes[0].grid(alpha=.2)
for i,(split,s) in enumerate(summaries.items()):
    values=[s['inside_percent']]+[s['within_percent'][str(t)] for t in [5,10,20,50]]
    axes[1].bar(np.arange(5)+(i-.5)*.36,values,.36,label=split)
axes[1].set_xticks(np.arange(5));axes[1].set_xticklabels(['Inside','<=5m','<=10m','<=20m','<=50m'])
axes[1].set(ylabel='Coverage among episodes with contours (%)',ylim=(0,100));axes[1].legend()
fig.tight_layout();fig.savefig(OUT/'coverage.png',dpi=160);plt.close(fig)
valid=sorted([r for r in rows if r['split']=='val_unseen' and 'distance_m' in r],key=lambda r:r['distance_m'])
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,q in zip(axes,[0,.5,.9]):
    r=valid[int(q*(len(valid)-1))]
    for contour in r['contours']:
        p=np.array(contour);ax.fill(p[:,0],p[:,1],alpha=.3,color='steelblue');ax.plot(*np.vstack([p,p[0]]).T,color='steelblue')
    ax.scatter(*r['target_xy'],c='red',marker='*',s=140,label='True target (scoring only)')
    ax.set_title(f"Quantile {q:.0%}: distance {r['distance_m']:.1f} m")
    ax.set_aspect('equal',adjustable='datalim');ax.margins(.2);ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)')
    ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'examples.png',dpi=160);plt.close(fig)
print(json.dumps(summaries,indent=2))
