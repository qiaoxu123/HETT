#!/usr/bin/env python3
"""Paired controls and gate decision on fixed val_seen selection."""
from __future__ import annotations
import json,sys,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.config import load_config,load_landmarks
from sensaturban_fpv.spatial_graph import build_block_graph
from sensaturban_fpv.spatial_graph.gate_f_programs import score_candidates
from scripts.eval_gate_f import metric,combine,aggregate
OUT=ROOT/'artifacts/spatial_canonicalization_gate_f'

def shuffled_program(p):
    if p.startswith(('BACK_','FRONT_')):return 'LEFT_'+p.split('_',1)[1]
    if p.startswith(('LEFT_','RIGHT_')):return 'FRONT_'+p.split('_',1)[1]
    return {'BETWEEN_TERNARY':'CENTER_DISTANCE','PAIRWISE_OLD':'BETWEEN_TERNARY',
            'DISTANCE_PRIOR':'BETWEEN_TERNARY','CENTER_DISTANCE':'FRONT_START',
            'FOOTPRINT_DISTANCE':'FRONT_START','ROAD_ASSOCIATION':'OFF_ROAD_BUFFER',
            'NEAR_ROAD_REGION':'OFF_ROAD_BUFFER','IN_ROAD_BUFFER':'OFF_ROAD_BUFFER',
            'OFF_ROAD_BUFFER':'IN_ROAD_BUFFER','ALONG_ROAD':'OFF_ROAD_BUFFER',
            'ALONG_ROAD_TANGENT':'OFF_ROAD_BUFFER'}.get(p,'CENTER_DISTANCE')

def paired(a):
    clusters=defaultdict(list)
    for entry in a:
        if entry['left'] is not None and entry['right'] is not None:
            clusters[(entry['source'],entry['episode_id'],entry['phrase'])].append(entry['left']-entry['right'])
    diffs=np.array([np.mean(v) for v in clusters.values()])
    wins=int((diffs>1e-9).sum());losses=int((diffs< -1e-9).sum());ties=int(len(diffs)-wins-losses)
    return {'n_clauses':len(a),'n_episode_clusters':len(diffs),'wins':wins,'losses':losses,'ties':ties,
            'mean_auc_difference':float(diffs.mean()) if len(diffs) else None,
            'paired_sign_p_one_sided':float(binomtest(wins,wins+losses,.5,alternative='greater').pvalue) if wins+losses else None}

def main():
    rows={r['sample_id']:r for r in (json.loads(x) for x in (OUT/'extended_geometry_rows.jsonl').open())}
    evaluated=[json.loads(x) for x in (OUT/'evaluated_scores.jsonl').open()]
    recs=json.loads((OUT/'program_induction.json').read_text())
    cal={(r['phrase'],r['anchor_type'],r['source']):r for r in recs if r['context']}
    phrase_cal={(r['phrase'],r['source']):r for r in recs if not r['context']}
    cfg=load_config();objects=load_landmarks(cfg);graphs={};cache={};paired_data=defaultdict(list);controls=[]
    for e in evaluated:
        key=(e['phrase'],e['kind'],e['source']);record=cal.get(key)
        if record is None:continue
        row=rows[e['id']];graph=graphs.get(row['map_id'])
        if graph is None:
            graph=build_block_graph(objects[row['map_id']],row['map_id']);graphs[row['map_id']]=graph
            print('graph',row['map_id'],flush=True)
        ev=row['evaluation'];target_id=int(ev['target_id']);ids=[int(x) for x in ev['candidate_ids'] if int(x) in graph.by_id and graph.by_id[int(x)].entity_type==graph.by_id[target_id].entity_type]
        # Use precisely the evaluator's candidate subset/order when measuring controls.
        from sensaturban_fpv.entity_geometry import group_of
        ids=[int(x) for x in ev['candidate_ids'] if int(x) in graph.by_id and group_of(graph.by_id[int(x)].entity_type)==group_of(graph.by_id[target_id].entity_type)]
        if len(ids)!=len(next(iter(e['scores'].values()))):continue
        candidates=[graph.by_id[i] for i in ids];target=ids.index(target_id)
        anchors=[graph.by_id[x['id']] for x in row['anchor_hypotheses'] if x['id'] in graph.by_id]
        seconds=[graph.by_id[x['id']] for x in row['second_anchor_hypotheses'] if x['id'] in graph.by_id]
        p=record['best_program'];correct=np.asarray(e['scores'].get(p)) if p in e['scores'] else None
        if correct is None:continue
        opposite=-correct if p.startswith(('BACK_','FRONT_','LEFT_','RIGHT_','CENTER_DISTANCE','FOOTPRINT_DISTANCE')) else None
        shuffle_rel=score_candidates(shuffled_program(p),row,candidates,anchors,seconds,graph,cache)
        alternative=[n for n in graph.named() if anchors and n.kind==anchors[0].kind and n.node_id not in {a.node_id for a in anchors}]
        alternative=sorted(alternative,key=lambda n:n.node_id)
        # Deterministic answer-blind shuffle; every sample gets one same-kind anchor.
        shuffled_anchor=alternative[0:1]
        shuffle_anchor=score_candidates(p,row,candidates,shuffled_anchor,seconds,graph,cache) if shuffled_anchor else None
        shuffled_second=[]
        if seconds:
            other=[n for n in graph.named() if n.kind==seconds[0].kind and n.node_id not in {x.node_id for x in seconds+anchors}]
            shuffled_second=sorted(other,key=lambda n:n.node_id)[:1]
        shuffle_second=score_candidates(p,row,candidates,anchors,shuffled_second,graph,cache) if shuffled_second else None
        cm=metric(correct,target)
        cases={'shuffled_relation':shuffle_rel,'opposite_relation':opposite,'shuffled_anchor':shuffle_anchor,'shuffled_second_anchor':shuffle_second}
        for label,sc in cases.items():
            if sc is None:continue
            m=metric(sc,target)
            if not m:continue
            paired_data[(e['split'],label)].append({'source':e['source'],'episode_id':e['episode_id'],'phrase':e['phrase'],'left':cm['auc'],'right':m['auc']})
            entry={'source':e['source'],'episode_id':e['episode_id'],'phrase':e['phrase'],'left':cm['auc'],'right':m['auc']}
            paired_data[(e['split'],e['phrase'],label)].append(entry)
            paired_data[(e['split'],e['phrase'],label,e['source'],e['protocol'])].append(entry)
        controls.append({'id':e['id'],'split':e['split'],'phrase':e['phrase'],'source':e['source'],
                         'correct_auc':cm['auc'],'controls':{label:metric(sc,target)['auc'] if sc is not None and metric(sc,target) else None for label,sc in cases.items()}})
    results={str(k):paired(v) for k,v in paired_data.items()}
    summary=json.loads((OUT/'gate_f_metrics.json').read_text());summary['controls']=results
    # F3 vs F0 and F4 vs F0 on exactly paired rows, with same tie treatment.
    variant_pairs=defaultdict(list);variant_pairs_by_source=defaultdict(list);context_pairs=defaultdict(list)
    for e in evaluated:
        rec=cal.get((e['phrase'],e['kind'],e['source']))
        if rec is None:continue
        row=rows[e['id']];target=e['target'];scores={p:np.asarray(s) for p,s in e['scores'].items()}
        baseline=np.asarray(e['old_scores']) if e.get('old_scores') is not None else np.zeros(len(next(iter(scores.values()))))
        base=metric(baseline,target)
        phrase_rec=phrase_cal.get((e['phrase'],e['source']))
        single=scores.get(phrase_rec['best_program']) if phrase_rec else None
        contextual=scores.get(rec['best_program'])
        if single is not None and contextual is not None:
            sm=metric(single,target);cm=metric(contextual,target)
            if sm and cm:context_pairs[(e['split'],e['source'],e['protocol'])].append({'source':e['source'],'episode_id':e['episode_id'],'phrase':e['phrase'],'left':cm['auc'],'right':sm['auc']})
        for name,sc in [('F1',single),('F2',contextual),('F3',combine(scores,rec['probabilities'])),('F4',combine(scores,rec['probabilities'],rec['reliability']))]:
            if sc is None or base is None:continue
            m=metric(sc,target)
            if m:
                entry={'source':e['source'],'episode_id':e['episode_id'],'phrase':e['phrase'],'left':m['auc'],'right':base['auc']}
                variant_pairs[(e['split'],name)].append(entry)
                variant_pairs_by_source[(e['split'],name,e['source'],e['protocol'])].append(entry)
    summary['paired_variants']={str(k):paired(v) for k,v in variant_pairs.items()}
    summary['paired_variants_by_source']={str(k):paired(v) for k,v in variant_pairs_by_source.items()}
    summary['context_vs_phrase']={str(k):paired(v) for k,v in context_pairs.items()}
    high=[r for r in recs if r['context'] and r['source']=='annotation' and r['n_val_seen']>=20]
    stable=[r for r in high if not r['unstable_seen_unseen']]
    # Gate requires all five stated conditions. Missing evidence counts against a pass.
    wins_shuffle=sum(1 for p in {r['phrase'] for r in high} if (z:=results.get(str(('val_unseen',p,'shuffled_relation','annotation','legacy_hard_negative')))) and z['paired_sign_p_one_sided'] is not None and z['paired_sign_p_one_sided']<.05 and z['mean_auc_difference']>0)
    wins_opposite=sum(1 for p in {r['phrase'] for r in high} if (z:=results.get(str(('val_unseen',p,'opposite_relation','annotation','legacy_hard_negative')))) and z['paired_sign_p_one_sided'] is not None and z['paired_sign_p_one_sided']<.05 and z['mean_auc_difference']>0)
    f3=summary['paired_variants_by_source'].get(str(('val_unseen','F3','annotation','legacy_hard_negative')))
    conditions={'three_classes_over_shuffle':wins_shuffle>=3,'two_classes_over_opposite':wins_opposite>=2,
                'probabilistic_over_old':bool(f3 and f3['mean_auc_difference']>0 and f3['paired_sign_p_one_sided'] is not None and f3['paired_sign_p_one_sided']<.05),
                'no_systemic_sign_flip':len(stable)==len(high),
                'view_dependent_stable_or_unknown':all(r['reliability']==0 or not r['unstable_seen_unseen'] for r in high if r['phrase'] in ('behind','in front of','left of','right of'))}
    summary['gate_f']={'status':'PASS' if all(conditions.values()) else 'FAIL','conditions':conditions,
                       'high_frequency_contexts':len(high),'stable_contexts':len(stable),
                       'caveat':'F0 executes existing hard ontology satisfaction for near/between and is neutral for unsupported words; learned teacher/DeepSeek pipeline is not reproduced.'}
    (OUT/'gate_f_metrics.json').write_text(json.dumps(summary,indent=2))
    (OUT/'control_results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in controls))
    # True ternary versus explicitly requested controls, annotation path only.
    between=[e for e in evaluated if e['phrase']=='between' and e['source']=='annotation']
    between_stats={'counts':Counter(e['split'] for e in between),'program_metrics':{}}
    for split in ('train_seen','val_seen','val_unseen'):
        subset=[e for e in between if e['split']==split]
        between_stats['program_metrics'][split]={p:aggregate([e['metrics'][p] for e in subset if p in e['metrics']]) for p in ('DISTANCE_PRIOR','PAIRWISE_OLD','BETWEEN_TERNARY')}
    between_stats['counts']=dict(between_stats['counts'])
    between_stats['controls']={k:v for k,v in results.items() if 'between' in k}
    (OUT/'between_control_summary.json').write_text(json.dumps(between_stats,indent=2))
    print(summary['gate_f'])
if __name__=='__main__':main()
