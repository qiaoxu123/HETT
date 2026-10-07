#!/usr/bin/env python3
"""Controlled raw-corpus road clauses absent from the legacy 2,290-row sample."""
import hashlib,json,sys,random,re
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.config import load_config,load_landmarks
from sensaturban_fpv.citynav import load_split
from sensaturban_fpv.spatial_graph import build_block_graph
from sensaturban_fpv.spatial_graph.gate_f_binding import bind_phrase
from sensaturban_fpv.spatial_graph.gate_f_geometry import trajectory_frames,road_projection
from sensaturban_fpv.spatial_graph.semantic_canonicalizer import parse_clauses,anchor_type
from sensaturban_fpv.spatial_graph.node_types import NodeKind
from sensaturban_fpv.entity_geometry import group_of
OUT=ROOT/'artifacts/spatial_canonicalization_gate_f'
PHRASES=('on','off','along','across','across from')
PER_SPLIT=40

def choose_pool(graph,target,sample_id):
    group=group_of(target.entity_type)
    pool=[n.node_id for n in graph.nodes if n.kind.is_instance and group_of(n.entity_type)==group and n.node_id!=target.node_id]
    pool.sort(key=lambda x:hashlib.sha256(f'{sample_id}:{x}'.encode()).digest())
    ids=pool[:9]+[target.node_id]
    ids.sort(key=lambda x:hashlib.sha256(f'order:{sample_id}:{x}'.encode()).digest())
    return ids

def main():
    cfg=load_config();objects=load_landmarks(cfg);graphs={};counts=Counter();missing=Counter()
    output=[]
    for split in ('train_seen','val_seen','val_unseen'):
        episodes=list(load_split(Path(cfg['paths']['citynav_dir']),split))
        random.Random(20261007).shuffle(episodes)
        for ep in episodes:
            if not ep.object_ids or not ep.description:continue
            clauses=parse_clauses(ep.description)['clauses']
            wanted=[(j,c) for j,c in enumerate(clauses) if c['raw_phrase'] in PHRASES and anchor_type(c['anchor_phrase'])=='road' and counts[(split,c['raw_phrase'])]<PER_SPLIT]
            if not wanted:continue
            graph=graphs.get(ep.map_name)
            if graph is None:graph=build_block_graph(objects[ep.map_name],ep.map_name);graphs[ep.map_name]=graph
            target=graph.by_id.get(int(ep.object_ids[0]))
            if target is None or not target.kind.is_instance:continue
            frames=trajectory_frames(ep.trajectory)
            for j,c in wanted:
                road_phrase=c['anchor_phrase'];second_phrase=None
                if c['raw_phrase']=='across' and ' from ' in road_phrase.lower():
                    road_phrase,second_phrase=re.split(r'\s+from\s+',road_phrase,maxsplit=1,flags=re.I)
                names=bind_phrase(road_phrase,graph)
                roads=[n for n in names if n.kind is NodeKind.ROAD_REGION]
                second=[n for n in bind_phrase(second_phrase,graph) if n.kind.is_instance] if second_phrase else []
                if not roads:missing[(split,c['raw_phrase'])]+=1;continue
                road=roads[0] if len(roads)==1 else None
                sample_id=f'{split}:{ep.index}:rawroad:{j}'
                candidates=choose_pool(graph,target,sample_id)
                if len(candidates)<2:continue
                projection=None
                if road is not None:
                    ri=next(i for i,r in enumerate(graph.regions) if r.node_id==road.node_id)
                    projection=road_projection(ep.trajectory[0,:2],road,graph.context.road_members[ri])
                row={'sample_id':sample_id,'split':split,'map_id':ep.map_name,'episode_id':ep.index,
                     'annotation_id':ep.ann_ids[0] if ep.ann_ids else None,'instruction':ep.description,
                     'relation_phrase':c['raw_phrase'],'clause_source':'raw_parser',
                     'anchor_phrase':road_phrase,'anchor_type':'road_region' if road else None,
                     'anchor_entity_id':road.node_id if road else None,
                     'anchor_position':road.center.tolist() if road else None,'anchor_footprint':None,
                     'anchor_hypotheses':[{'id':n.node_id,'type':n.kind.value,'name':n.name,'position':n.center.tolist(),'footprint':None} for n in roads],
                     'second_anchor_phrase':second_phrase,'second_anchor_type':second[0].kind.value if len(second)==1 else None,
                     'second_anchor_entity_id':second[0].node_id if len(second)==1 else None,
                     'second_anchor_position':second[0].center.tolist() if len(second)==1 else None,
                     'second_anchor_footprint':second[0].footprint.tolist() if len(second)==1 else None,
                     'second_anchor_hypotheses':[{'id':n.node_id,'type':n.kind.value,'name':n.name,'position':n.center.tolist(),'footprint':n.footprint.tolist()} for n in second],
                     'target_type':target.kind.value,'target_position':target.center.tolist(),
                     'target_footprint':target.footprint.tolist(),**frames,
                     'road_region_id':road.node_id if road else None,'road_region_name':road.name if road else None,
                     'road_nearest_point':projection.nearest_point.tolist() if projection else None,
                     'road_tangent':projection.tangent.tolist() if projection else None,
                     'road_normal':projection.normal.tolist() if projection else None,
                     'road_side_sign':int(np.sign(projection.side)) if projection else None,
                     'anchor_major_axis':None,'anchor_minor_axis':None,
                     'evaluation':{'candidate_ids':candidates,'target_index':candidates.index(target.node_id),'target_id':target.node_id,
                                   'candidate_protocol':'9 deterministic map distractors plus target; separate from legacy hard-negative protocol'}}
                output.append(row);counts[(split,c['raw_phrase'])]+=1
            if all(counts[(split,p)]>=PER_SPLIT for p in PHRASES):break
        print(split,{p:counts[(split,p)] for p in PHRASES},flush=True)
    with (OUT/'extended_geometry_rows.jsonl').open('a') as file:
        for row in output:file.write(json.dumps(row)+'\n')
    (OUT/'road_corpus_sampling.json').write_text(json.dumps({'per_split_limit':PER_SPLIT,'rows_added':len(output),
      'counts':{f'{s}:{p}':n for (s,p),n in counts.items()},'unbound_attempts':{f'{s}:{p}':n for (s,p),n in missing.items()},
      'candidate_protocol':'separate deterministic 9 map distractors plus target; not pooled with legacy hard-negative metrics'},indent=2))
    print('added',len(output))
if __name__=='__main__':main()
