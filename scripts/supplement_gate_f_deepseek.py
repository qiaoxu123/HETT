#!/usr/bin/env python3
"""Execute cached DeepSeek semantic clauses on the same answer-blind map binder."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.config import load_config,load_landmarks
from sensaturban_fpv.spatial_graph import build_block_graph
from sensaturban_fpv.spatial_graph.gate_f_binding import bind_phrase,text_between_anchors
from sensaturban_fpv.spatial_graph.gate_f_geometry import anchor_axes,road_projection
from sensaturban_fpv.spatial_graph.node_types import NodeKind
from sensaturban_fpv.spatial_graph.semantic_canonicalizer import normalize_relation_phrase
OUT=ROOT/'artifacts/spatial_canonicalization_gate_f'

def node_json(n):
    return {'id':n.node_id,'type':n.kind.value,'name':n.name,'position':n.center.tolist(),
            'footprint':n.footprint.tolist() if n.kind is not NodeKind.ROAD_REGION else None}

def main():
    validation=json.loads((OUT/'parser_validation.json').read_text())
    existing=[json.loads(x) for x in (OUT/'extended_geometry_rows.jsonl').open()]
    template={}
    for r in existing:
        if r['clause_source']=='annotation':template.setdefault(r['instruction'],r)
    cfg=load_config();objects=load_landmarks(cfg);graphs={};added=[];unmatched=0
    for check in validation['records']:
        parsed=check.get('prompt_a');source=template.get(check['instruction'])
        if not parsed or source is None:unmatched+=1;continue
        graph=graphs.get(source['map_id'])
        if graph is None:graph=build_block_graph(objects[source['map_id']],source['map_id']);graphs[source['map_id']]=graph
        for j,clause in enumerate(parsed['clauses']):
            phrase=normalize_relation_phrase(clause['raw_phrase'],clause.get('semantic_family'));first=clause['anchor_phrase'];second_phrase=None
            if phrase=='between' and (pair:=text_between_anchors(source['instruction'])):
                first,second_phrase=pair
            anchors=bind_phrase(first,graph)
            seconds=bind_phrase(second_phrase,graph) if second_phrase else []
            road_nodes=[n for n in anchors if n.kind is NodeKind.ROAD_REGION]
            road=road_nodes[0] if len(road_nodes)==1 else None
            unique=anchors[0] if len(anchors)==1 else None
            major,minor=anchor_axes(unique.footprint) if unique and unique.kind is not NodeKind.ROAD_REGION else (None,None)
            projection=None
            if road:
                index=next(i for i,r in enumerate(graph.regions) if r.node_id==road.node_id)
                projection=road_projection(source['start_position'],road,graph.context.road_members[index])
            row=dict(source)
            row.update({'sample_id':f"{source['split']}:{source['episode_id']}:deepseek:{j}",
                        'clause_source':'deepseek','relation_phrase':phrase,'semantic_clause':clause,
                        'anchor_phrase':first,'anchor_type':unique.kind.value if unique else None,
                        'anchor_entity_id':unique.node_id if unique else None,
                        'anchor_position':unique.center.tolist() if unique else None,
                        'anchor_footprint':unique.footprint.tolist() if unique and unique.kind is not NodeKind.ROAD_REGION else None,
                        'anchor_hypotheses':[node_json(n) for n in anchors],
                        'second_anchor_phrase':second_phrase,'second_anchor_type':seconds[0].kind.value if len(seconds)==1 else None,
                        'second_anchor_entity_id':seconds[0].node_id if len(seconds)==1 else None,
                        'second_anchor_position':seconds[0].center.tolist() if len(seconds)==1 else None,
                        'second_anchor_footprint':seconds[0].footprint.tolist() if len(seconds)==1 and seconds[0].kind is not NodeKind.ROAD_REGION else None,
                        'second_anchor_hypotheses':[node_json(n) for n in seconds],
                        'road_region_id':road.node_id if road else None,'road_region_name':road.name if road else None,
                        'road_nearest_point':projection.nearest_point.tolist() if projection else None,
                        'road_tangent':projection.tangent.tolist() if projection else None,
                        'road_normal':projection.normal.tolist() if projection else None,
                        'road_side_sign':int(np.sign(projection.side)) if projection else None,
                        'anchor_major_axis':major.tolist() if major is not None else None,
                        'anchor_minor_axis':minor.tolist() if minor is not None else None})
            added.append(row)
    with (OUT/'extended_geometry_rows.jsonl').open('a') as file:
        for row in added:file.write(json.dumps(row)+'\n')
    (OUT/'deepseek_geometry_path.json').write_text(json.dumps({'instructions_in_validation':len(validation['records']),
          'unmatched_in_legacy_samples':unmatched,'clause_rows_added':len(added),
          'bound_anchor_rows':sum(bool(r['anchor_hypotheses']) for r in added),
          'canonical_unknown_clauses':sum(r['relation_phrase']=='UNKNOWN' for r in added),
          'note':'DeepSeek supplied text roles only; program probabilities come from geometry calibration.'},indent=2))
    print(len(added),unmatched)
if __name__=='__main__':main()
