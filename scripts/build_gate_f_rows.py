#!/usr/bin/env python3
"""Extend legacy diagnostic rows with verified CityNav/CityRefer map geometry."""
import json,sys,os
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.config import load_config,load_landmarks
from sensaturban_fpv import citynav
from sensaturban_fpv.spatial_graph import build_block_graph
from sensaturban_fpv.spatial_graph.gate_f_binding import annotation_clauses,parser_clauses,bind_phrase
from sensaturban_fpv.spatial_graph.gate_f_geometry import trajectory_frames,anchor_axes,road_projection
from sensaturban_fpv.spatial_graph.node_types import NodeKind
SRC=Path(os.environ.get('GATE_F_SOURCE_ROWS', ROOT.parent/'Achieved/hett-experiments/55-sensaturban-fpv-rendering/artifacts/relation_v2/relation_samples.jsonl'))
OUT=ROOT/'artifacts/spatial_canonicalization_gate_f'

def node_dict(n):
    if n is None:return None
    return {'id':n.node_id,'type':n.kind.value,'name':n.name,'position':n.center.tolist(),
            'footprint':n.footprint.tolist() if n.kind is not NodeKind.ROAD_REGION else None}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=load_config();objects=load_landmarks(cfg);samples=[json.loads(x) for x in SRC.open()]
    needed=defaultdict(set)
    for r in samples:needed[r['split']].add(r['episode_index'])
    episodes={}
    for split,indices in needed.items():
        ep=citynav.load_split(Path(cfg['paths']['citynav_dir']),split)
        episodes.update({(split,i):ep[i] for i in indices})
    graphs={};stats=Counter();trajectory_stats=defaultdict(list);road_stats=Counter()
    with (OUT/'extended_geometry_rows.jsonl').open('w') as file:
        for number,s in enumerate(samples):
            ep=episodes[(s['split'],s['episode_index'])];graph=graphs.get(s['map'])
            if graph is None:
                graph=build_block_graph(objects[s['map']],s['map']);graphs[s['map']]=graph
                print('graph',s['map'],len(graph.nodes),len(graph.regions),flush=True)
            target=graph.by_id.get(int(s['target_id']))
            if target is None:stats['missing_target']+=1;continue
            frames=trajectory_frames(ep.trajectory)
            for n,v in frames['heading_stability'].items():
                if v is not None:trajectory_stats[n].append(v)
            clauses=annotation_clauses(s)+parser_clauses(s)
            for clause in clauses:
                anchors=bind_phrase(clause['anchor_phrase'],graph)
                second=bind_phrase(clause['second_anchor_phrase'],graph) if clause['second_anchor_phrase'] else []
                road_nodes=[n for n in anchors if n.kind is NodeKind.ROAD_REGION]
                road=road_nodes[0] if len(road_nodes)==1 else None
                if road is not None:
                    idx=graph.regions.index(road)
                    proj=road_projection(road.center,road,graph.context.road_members[idx])
                    road_info={'id':road.node_id,'name':road.name,'nearest_point':proj.nearest_point.tolist(),
                               'tangent':proj.tangent.tolist(),'normal':proj.normal.tolist(),
                               'side_sign':int(np.sign(proj.side))}
                    road_stats['named_road_bound']+=1
                else:road_info=None
                first=anchors[0] if len(anchors)==1 else None
                major,minor=anchor_axes(first.footprint) if first is not None and first.kind is not NodeKind.ROAD_REGION else (None,None)
                row={'sample_id':f"{s['split']}:{s['episode_index']}:{s['step']}:{clause['source']}:{clause['clause_index']}",
                     'split':s['split'],'map_id':s['map'],'episode_id':s['episode_index'],
                     'annotation_id':ep.ann_ids[0] if ep.ann_ids else None,'instruction':s['instruction'],
                     'relation_phrase':clause['phrase'],'clause_source':clause['source'],
                     'anchor_phrase':clause['anchor_phrase'],'anchor_type':first.kind.value if first else None,
                     'anchor_entity_id':first.node_id if first else None,
                     'anchor_position':first.center.tolist() if first else None,
                     'anchor_footprint':first.footprint.tolist() if first and first.kind is not NodeKind.ROAD_REGION else None,
                     'anchor_hypotheses':[node_dict(n) for n in anchors],
                     'second_anchor_phrase':clause['second_anchor_phrase'],
                     'second_anchor_type':second[0].kind.value if len(second)==1 else None,
                     'second_anchor_entity_id':second[0].node_id if len(second)==1 else None,
                     'second_anchor_position':second[0].center.tolist() if len(second)==1 else None,
                     'second_anchor_footprint':second[0].footprint.tolist() if len(second)==1 and second[0].kind is not NodeKind.ROAD_REGION else None,
                     'second_anchor_hypotheses':[node_dict(n) for n in second],
                     'target_type':target.kind.value,'target_position':target.center.tolist(),
                     'target_footprint':target.footprint.tolist(),
                     **frames,
                     'road_region_id':road_info['id'] if road_info else None,
                     'road_region_name':road_info['name'] if road_info else None,
                     'road_nearest_point':road_info['nearest_point'] if road_info else None,
                     'road_tangent':road_info['tangent'] if road_info else None,
                     'road_normal':road_info['normal'] if road_info else None,
                     'road_side_sign':road_info['side_sign'] if road_info else None,
                     'anchor_major_axis':major.tolist() if major is not None else None,
                     'anchor_minor_axis':minor.tolist() if minor is not None else None,
                     # Offline evaluator only. Neither parser nor frame construction accepts these keys.
                     'evaluation':{'candidate_ids':s['candidate_ids'],'target_index':s['target_index'],'target_id':s['target_id']}}
                file.write(json.dumps(row)+'\n')
                stats['clauses']+=1;stats['anchor_bound']+=int(bool(anchors));stats['unique_anchor']+=int(len(anchors)==1)
                stats['between_explicit_second']+=int(clause['phrase']=='between' and bool(second))
                stats['road_bound']+=int(bool(road_nodes))
            stats['samples']+=1
    (OUT/'trajectory_frame_stats.json').write_text(json.dumps({'samples':stats['samples'],'heading_stability':{n:{'count':len(v),'mean':float(np.mean(v))} for n,v in trajectory_stats.items()}},indent=2))
    (OUT/'road_geometry_stats.json').write_text(json.dumps({'graphs':len(graphs),'regions':sum(len(g.regions) for g in graphs.values()),'counts':dict(road_stats)},indent=2))
    (OUT/'row_build_stats.json').write_text(json.dumps(dict(stats),indent=2))
    print(dict(stats))
if __name__=='__main__':main()
