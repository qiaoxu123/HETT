#!/usr/bin/env python3
"""Render the measured Gate G results, including all abstentions."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/instruction_defined_layout'
def load(name):return json.loads((OUT/name).read_text())
def pct(x):return f'{x:.1%}'
def num(x):return '—' if x is None else f'{x:.3f}'
def matrix(metrics,split):
    lines=['| Arm | n | Top1 | Top4 | MRR | Pairwise AUC | Mean rank |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,label in [('L0','Anchor prior'),('L1','Global layout'),('L2','PCA layout'),('L3','Instruction frame'),('L4','Instruction + anchor'),('L5','Soft ordinal'),('L6','Shuffled index'),('L7','Flipped frame'),('L8','Shuffled reference')]:
        r=metrics['splits'][split][arm];lines.append(f"| {arm} {label} | {r['n']} | {num(r.get('top1'))} | {num(r.get('top4'))} | {num(r.get('mrr'))} | {num(r.get('auc'))} | {num(r.get('rank'))} |")
    return '\n'.join(lines)
def paired(rows,split,a,b):
    sample=[r for r in rows if r['split']==split and r['metrics'].get(a) and r['metrics'].get(b)]
    diff=[r['metrics'][a]['auc']-r['metrics'][b]['auc'] for r in sample]
    w=sum(x>1e-9 for x in diff);l=sum(x<-1e-9 for x in diff)
    return {'n':len(diff),'wins':w,'losses':l,'ties':len(diff)-w-l,'delta':float(np.mean(diff)) if diff else None,
            'p':float(binomtest(w,w+l,.5,alternative='greater').pvalue) if w+l else None}
def main():
    audit=load('layout_language_audit.json');metrics=load('gate_g_metrics.json');parser=load('parser_validation.json');scores=load('layout_scores.json');road=load('road_sequences.json');cf=load('layout_counterfactuals.json')
    counts=audit['counts'];total=counts['instructions'];frame_count=counts['frame'];layout_count=counts['layout'];both=counts['frame_and_layout']
    frame_counts={s:sum(r['split']==s for r in load('local_frames.json')) for s in ('train_seen','val_seen','val_unseen')}
    report=['# Instruction-defined frame diagnostic','',
            'A frame is built from instruction text and map objects before the offline target ID is read. Geometry evaluation uses the deterministic rule parser; DeepSeek is evaluated separately for language consistency. The executable parking scope is selected from a uniquely named reference or a unique parking region. If several parking regions remain, the sample abstains. The true car is never inserted into the candidate set after scope selection.','',
            f"Detected frame cues: {frame_count:,}/{total:,} ({pct(frame_count/total)}). Detected frame plus ordinal layout: {both:,}/{total:,} ({pct(both/total)}). This is regex coverage, not parser precision.",'',
            f"DeepSeek sampled {parser['n']} instructions with two ontology orders; {parser['valid_both']} produced valid JSON both times. Exact consistency {pct(parser['exact'])}; frame fields {pct(parser['frame_exact'])}; reference phrase {pct(parser['reference_exact'])}; layout Jaccard {num(parser['operator_jaccard'])}; ordinal agreement {pct(parser['ordinal_exact'])}.",'',
            '| Split | Parsed frame + layout | Resolved oriented frame in evaluated parking scope |','|---|---:|---:|']
    for s in ('train_seen','val_seen','val_unseen'):report.append(f"| {s} | {counts[s+'_frame_and_layout']} | {frame_counts[s]} |")
    report+=['','A building footprint major/minor axis is *unoriented*. The words “long side to the right” alone do not mathematically identify which physical end of that axis is right. The implementation refuses to choose a sign from GT or an arbitrary target position. A separate located reference (“road at the top”, “building to the right”) can resolve the sign relative to the text-selected parking region. Near-square footprints are rejected at axis confidence < 0.12.','',
             'PCA and global frames are evaluated as baselines. The 180° flip and reference swap are behavioral controls; a change in score demonstrates execution sensitivity, not grounding accuracy.','']
    (ROOT/'INSTRUCTION_DEFINED_FRAME_REPORT.md').write_text('\n'.join(report).rstrip()+'\n')
    statuses=metrics['status'];parking=['# Parking layout reasoning','',
       'Cars are collected only inside a text-selected CityRefer Parking polygon (2 m tolerance); negatives are all other cars in that same lot. Grouping and ordering use map centers before the answer is consulted. This is a hard same-lot candidate set, not random negatives. It can still contain multiple physical subgrids.','',
       f"Evaluated {statuses.get('evaluated',0)} samples. Abstentions: ambiguous parking region {statuses.get('ambiguous_parking_region',0)}, selected region misses GT {statuses.get('scope_miss',0)}, too few cars {statuses.get('too_few_cars',0)}, non-car target {statuses.get('non_car_target',0)}.",'',
       '## Val seen','',matrix(metrics,'val_seen'),'','## Val unseen','',matrix(metrics,'val_unseen'),'']
    for split in ('val_seen','val_unseen'):
        family=metrics['families'][split]
        parking+=['',f'## {split} layout families','', '| Family | Global n / Top1 / MRR / rank | Instruction n / Top1 / MRR / rank |','|---|---|---|']
        for name,r in sorted(family.items(),key=lambda x:-x[1]['L1']['n']):
            if r['L1']['n']==0:continue
            def cell(v):return f"{v['n']} / {num(v.get('top1'))} / {num(v.get('mrr'))} / {num(v.get('rank'))}"
            parking.append(f"| {name} | {cell(r['L1'])} | {cell(r['L3'])} |")
    for split in ('val_seen','val_unseen'):
        hard=[r['same_row_auc'] for r in scores if r['split']==split and r['same_row_auc'] is not None]
        parking+=['',f"Same-row neighbor AUC ({split}): {num(float(np.mean(hard)) if hard else None)} over {len(hard)} evaluable clauses. This compares the GT with cars assigned to the same inferred row."]
    parking+=['','Rows and columns use PCA or a resolved instruction frame followed by gap clustering. This first pass has no validated lot subdivision, so very large parking polygons can create incorrect row numbers. L1/L2 assume bottom-first row order as a baseline; L3 applies ROW_INDEX only when the instruction states its row start, otherwise that clause abstains.']
    (ROOT/'PARKING_LAYOUT_REASONING_REPORT.md').write_text('\n'.join(parking).rstrip()+'\n')
    road_lines=['# Road sequence reasoning','',f"Attempted {road['attempted']} text-detected road sequence instructions. Status: {road['status']}.",'',
       'A named road must resolve to one connected RoadRegion. Candidate objects are selected by text-inferred class within 15 m of that region. A curvilinear coordinate is computed from projection onto the region polyline. The GT is consulted only after the candidate set exists.','',
       'A road polyline has two ends. “First”, “last”, or “from the corner” cannot choose an end without a text-bound start/corner. These clauses abstain; selecting the end nearest the GT would leak the answer. HALFWAY_ALONG is orientation invariant and can be evaluated when a named road and candidate class bind.','',
       '| Split | Evaluable halfway clauses | Top1 | MRR | AUC |','|---|---:|---:|---:|---:|']
    for s in ('train_seen','val_seen','val_unseen'):
        rs=[r for r in road['rows'] if r['split']==s]
        road_lines.append(f"| {s} | {len(rs)} | {num(np.mean([r['top1'] for r in rs]) if rs else None)} | {num(np.mean([r['mrr'] for r in rs]) if rs else None)} | {num(np.mean([r['auc'] for r in rs]) if rs else None)} |")
    (ROOT/'ROAD_SEQUENCE_REASONING_REPORT.md').write_text('\n'.join(road_lines).rstrip()+'\n')
    comparisons=[('L3 vs L1','L3','L1'),('L3 vs shuffled ordinal','L3','L6'),('L3 vs flipped frame','L3','L7'),('L3 vs shuffled reference','L3','L8')]
    gate=['# Explicit layout Gate G','', '**Gate G: FAIL under the specified conjunction.** The failure is a coverage and grounding limitation, not proof that all layout language lacks information. No visual fusion was attempted.','',
          f"Full corpus: {total:,} instructions; explicit layout {layout_count:,} ({pct(layout_count/total)}); explicit frame cue {frame_count:,} ({pct(frame_count/total)}); both {both:,} ({pct(both/total)}). True target-type counts for layout instructions: "+', '.join(f"{k[14:]} {v:,}" for k,v in sorted(counts.items()) if k.startswith('layout_target_'))+'.',
          f"Group A/B/C/other text counts: {counts['group_A']:,}/{counts['group_B']:,}/{counts['group_C']:,}/{counts['group_OTHER']:,}. Group C has generic relations but no explicit layout/frame and is not scored by L0–L8. On val_unseen Group A has only {counts['val_unseen_frame_and_layout']} detected instructions; the evaluable L3 count is shown below.",'',
          '## Ablations: val seen','',matrix(metrics,'val_seen'),'','## Ablations: val unseen','',matrix(metrics,'val_unseen'),'','The rows above have different coverage. Geometry scores use the rule parser; DeepSeek is a separate consistency diagnostic. Only the following matched-pair tests support direct arm comparisons. All p-values are one-sided exact sign tests over episode clauses; no val_unseen result was used to select a frame, threshold, or weight.','',
          '| Split | Pair | n | Wins / losses / ties | Mean AUC Δ | p |','|---|---|---:|---:|---:|---:|']
    for split in ('val_seen','val_unseen'):
        for label,a,b in comparisons:
            r=paired(scores,split,a,b)
            gate.append(f"| {split} | {label} | {r['n']} | {r['wins']}/{r['losses']}/{r['ties']} | {num(r['delta'])} | {num(r['p'])} |")
    changed={k:sum(bool(r[k]) for r in cf) for k in ('index_changes','frame_changes','reference_changes')}
    gate+=['',f"Behavioral controls over {len(cf)} resolved-frame rows: index changes scores in {changed['index_changes']}, 180° frame flip in {changed['frame_changes']}, reference replacement in {changed['reference_changes']}. These are behavior checks, not significance tests.",'',
           f"Gate criteria fail because: (1) val_unseen has only {metrics['splits']['val_unseen']['L3']['n']} resolved Group A L3 parking rows, so stable transfer cannot be established; (2) ordinal shuffle does not consistently lose on val_seen; (3) two high-frequency families cannot be validated on unseen hard negatives; (4) DeepSeek parser exact agreement is low; (5) most candidate scopes are ambiguous or miss GT. The primary failure categories are parser instability, frame sign ambiguity, object grouping/scope binding, ordinal start ambiguity, and insufficient unseen coverage.",'',
           'The parking candidate set is same-lot and includes row neighbors. Road first/last/corner clauses abstain when the start end is unbound. These abstentions prevent a misleading oracle-selected success.']
    (ROOT/'EXPLICIT_LAYOUT_GATE_G_REPORT.md').write_text('\n'.join(gate).rstrip()+'\n')
    selective=['# Selective spatial constraint graph','',
       'This round implements the layout channel separately from anchor proximity and adds a small allowed-edge filter and two-channel score combiner. No graph fusion or weight tuning was run because Gate G failed. The allowed evidence-supported spatial edges remain NEAR, NEXT_TO, ROAD_ASSOCIATION, FOOTPRINT_DISTANCE, LAYOUT_ORDER, ROW_INDEX, COLUMN_INDEX, and ROAD_SEQUENCE. Generic BEHIND, FRONT, and implicit LEFT/RIGHT require an explicit frame; they remain disabled otherwise, as concluded by Gate F.','',
       'L4 uses a fixed, untuned 0.5/0.5 sum of layout and distance to a text-bound reference within the parking scope; its results are in the Gate G table. No val_unseen tuning occurred.']
    (ROOT/'SELECTIVE_SPATIAL_CONSTRAINT_REPORT.md').write_text('\n'.join(selective).rstrip()+'\n')
    print('Reports written')
if __name__=='__main__':main()
