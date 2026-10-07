#!/usr/bin/env python3
"""Road sequence diagnostic with named RoadRegion binding; no GT-chosen branch."""
import json,re,sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.config import load_config,load_landmarks
from sensaturban_fpv.spatial_graph.node_types import Node,NodeKind
from sensaturban_fpv.spatial_graph.road_region import build_regions,region_polyline
from sensaturban_fpv.spatial_graph.gate_f_geometry import road_projection
from sensaturban_fpv.spatial_graph.layout_program import road_s,road_sequence_score
OUT=ROOT/'artifacts/instruction_defined_layout'
def source_class(text):
    head=text.lower()[:110]
    matches=[(m.start(),'Car') for m in re.finditer(r'\b(?:car|vehicle|van)\b',head)]
    matches += [(m.start(),'Building') for m in re.finditer(r'\b(?:building|house|shop|church)\b',head)]
    return min(matches)[1] if matches else None
def roads(objects):
    segments=[]
    for o in objects.values():
        if o.object_type!='TrafficRoad' or not o.name:continue
        segments.append(Node(o.id,NodeKind.ROAD_SEGMENT,o.object_type,o.name,np.asarray(o.position,float),np.asarray(o.dimension,float),np.asarray(o.contour,float),np.array([1.,0.])))
    regions=build_regions(segments);members={r.node_id:[s for s in segments if s.node_id in r.member_ids] for r in regions}
    return regions,members
def main():
    cfg=load_config();objects=load_landmarks(cfg);parsed=[json.loads(x) for x in (OUT/'parsed_layout_programs.jsonl').open()]
    rows=[r for r in parsed if any(q['type'] in ('HALFWAY_ALONG','NTH_FROM_CORNER','ROAD_SEQUENCE_ORDER') for q in r['layout_programs'])]
    episodes={}
    for split in ('train_seen','val_seen','val_unseen'):
        data=json.loads((Path(cfg['paths']['citynav_dir'])/f'citynav_{split}.json').read_text())
        for r in rows:
            if r['split']==split:episodes[(split,r['episode_index'])]=data[r['episode_index']]
    cache={};out=[];status=Counter()
    for row in rows:
        ep=episodes[(row['split'],row['episode_index'])];map_id=f"{ep['area']}_block_{ep['block']}";objs=objects.get(map_id,{})
        if map_id not in cache:cache[map_id]=roads(objs)
        regions,members=cache[map_id];text=row['instruction'].lower();matches=[r for r in regions if len(r.name)>4 and r.name.lower() in text]
        if len(matches)!=1:status['road_unresolved']+=1;continue
        road=matches[0];klass=source_class(row['instruction'])
        if klass is None:status['target_class_unresolved']+=1;continue
        candidates=[o for o in objs.values() if o.object_type==klass and road_projection(o.position,road,members[road.node_id]).distance<=15]
        if len(candidates)<3:status['scope_too_small']+=1;continue
        program=next(q for q in row['layout_programs'] if q['type'] in ('HALFWAY_ALONG','NTH_FROM_CORNER','ROAD_SEQUENCE_ORDER'))
        if program['type']!='HALFWAY_ALONG':
            status['start_or_corner_unresolved']+=1;continue
        polyline,_=region_polyline(members[road.node_id]);s=road_s([o.position for o in candidates],polyline)
        scores=road_sequence_score(program,s);target_ids=ep.get('object_ids') or []
        if len(target_ids)!=1:status['target_record_unresolved']+=1;continue
        raw=target_ids[0];target=int(raw.get('object_id',raw.get('id',next(iter(raw.values())))) if isinstance(raw,dict) else raw)
        ids=[o.id for o in candidates]
        if target not in ids:status['scope_miss']+=1;continue
        ix=ids.index(target);order=np.argsort(-scores);rank=int(np.flatnonzero(order==ix)[0])+1
        out.append({'split':row['split'],'episode_index':row['episode_index'],'program':program['type'],'road_region':road.name,'n_candidates':len(ids),'rank':rank,'top1':rank==1,'mrr':1/rank,'auc':float(np.mean(scores[ix]>np.delete(scores,ix))+.5*np.mean(scores[ix]==np.delete(scores,ix)))})
        status['evaluated']+=1
    summary={'attempted':len(rows),'status':dict(status),'rows':out}
    (OUT/'road_sequences.json').write_text(json.dumps(summary,indent=2));print({'attempted':len(rows),'status':dict(status)})
if __name__=='__main__':main()
