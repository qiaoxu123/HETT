#!/usr/bin/env python3
"""Explicit B0-B4 on exactly the same two-anchor between clauses."""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.config import load_config,load_landmarks
from sensaturban_fpv.spatial_graph import build_block_graph
from sensaturban_fpv.spatial_graph.gate_f_programs import score_candidates
from sensaturban_fpv.entity_geometry import group_of
from scripts.eval_gate_f import metric,aggregate
from scripts.gate_f_controls import paired
OUT=ROOT/'artifacts/spatial_canonicalization_gate_f'

def main():
    rows=[json.loads(x) for x in (OUT/'extended_geometry_rows.jsonl').open()]
    rows=[r for r in rows if r['relation_phrase']=='between' and r['clause_source']=='annotation' and r['anchor_hypotheses'] and r['second_anchor_hypotheses']]
    cfg=load_config();objects=load_landmarks(cfg);graphs={};cache={};metrics=defaultdict(lambda:defaultdict(list));pairs=defaultdict(list);count=defaultdict(int)
    for r in rows:
        graph=graphs.get(r['map_id'])
        if graph is None:graph=build_block_graph(objects[r['map_id']],r['map_id']);graphs[r['map_id']]=graph
        ev=r['evaluation'];target_id=int(ev['target_id'])
        ids=[int(i) for i in ev['candidate_ids'] if int(i) in graph.by_id and group_of(graph.by_id[int(i)].entity_type)==group_of(graph.by_id[target_id].entity_type)]
        if len(ids)<2 or target_id not in ids:continue
        cand=[graph.by_id[i] for i in ids];target=ids.index(target_id)
        anchors=[graph.by_id[x['id']] for x in r['anchor_hypotheses'] if x['id'] in graph.by_id]
        seconds=[graph.by_id[x['id']] for x in r['second_anchor_hypotheses'] if x['id'] in graph.by_id]
        other=sorted([n for n in graph.named() if n.kind==seconds[0].kind and n.node_id not in {a.node_id for a in anchors+seconds}],key=lambda n:n.node_id) if seconds else []
        programs={'B0_distance':('DISTANCE_PRIOR',anchors,seconds),
                  'B1_pairwise':('PAIRWISE_OLD',anchors,seconds),
                  'B2_ternary':('BETWEEN_TERNARY',anchors,seconds),
                  'B3_shuffle_second':('BETWEEN_TERNARY',anchors,other[:1]),
                  'B4_shuffle_relation':('CENTER_DISTANCE',anchors,seconds)}
        got={}
        for name,(program,a,b) in programs.items():
            sc=score_candidates(program,r,cand,a,b,graph,cache)
            if sc is not None:
                got[name]=metric(sc,target)
                if got[name]:metrics[r['split']][name].append(got[name])
        if 'B2_ternary' in got:
            count[r['split']]+=1
            for other_name in ('B0_distance','B1_pairwise','B3_shuffle_second','B4_shuffle_relation'):
                if other_name in got:
                    pairs[(r['split'],other_name)].append({'source':'annotation','episode_id':r['episode_id'],'phrase':'between',
                                                            'left':got['B2_ternary']['auc'],'right':got[other_name]['auc']})
    result={'n_explicit_bound_clauses':dict(count),
            'program_metrics':{s:{p:aggregate(v) for p,v in data.items()} for s,data in metrics.items()},
            'B2_paired_comparisons':{str(k):paired(v) for k,v in pairs.items()}}
    (OUT/'between_ternary_stats.json').write_text(json.dumps(result,indent=2))
    print(result)
if __name__=='__main__':main()
