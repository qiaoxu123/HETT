#!/usr/bin/env python3
"""Human-readable, evidence-linked Gate F reports from frozen artifacts."""
import json
import ast,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from sensaturban_fpv.spatial_graph.semantic_canonicalizer import PHRASES;A=R/'artifacts/spatial_canonicalization_gate_f'
def load(name):return json.loads((A/name).read_text())
def num(x):return '—' if x is None else f'{x:.3f}'
def pnum(x):return '—' if x is None else (f'{x:.2e}' if x<.001 else f'{x:.3f}')
def auc(rec,program,split):
    x=rec[split].get(program)
    return num(x['auc']) if x else '—'
def table(lines,head,rows):
    lines.extend(['','| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|'])
    lines.extend('| '+' | '.join(str(x) for x in row)+' |' for row in rows)

def main():
    audit=load('input_audit.json');rows=load('row_build_stats.json');road=load('road_geometry_stats.json')
    parser=load('parser_validation.json');programs=load('program_induction.json');metrics=load('gate_f_metrics.json');between=load('between_ternary_stats.json')
    raw=load('road_corpus_sampling.json');trajectory=load('trajectory_frame_stats.json');deepseek=load('deepseek_geometry_path.json');coverage=load('row_coverage.json')
    # Input audit: retain existing checked joins, add measured extended-row coverage.
    inp=(R/'GATE_F_INPUT_AUDIT.md').read_text().split('## Extended rows')[0].rstrip().splitlines()
    inp+=['','## Extended rows',f"The 2,290 legacy samples produced {rows['clauses']:,} annotation/parser clause rows before supplementation. {rows['anchor_bound']:,} bound at least one anchor; {rows['road_bound']:,} bound a named RoadRegion; {rows['between_explicit_second']:,} between clause rows had a text-supported second anchor. These counts include both parsing paths.",
          f"A separate, randomized raw-corpus road sample added {raw['rows_added']:,} rows with deterministic map distractors. Its ranking difficulty differs from the legacy candidate lists, so its metrics are reported separately.",
          f"The controlled DeepSeek sample added {deepseek['clause_rows_added']:,} text-derived clause rows; {deepseek['bound_anchor_rows']:,} bound at least one named map hypothesis. These share the legacy candidate protocol and are reported as a separate source.",
          f"All paths together contain {coverage['counts']['rows']:,} clause rows from {coverage['unique_episodes']:,} unique episodes: {coverage['counts']['trajectory_rows']:,} carry trajectory, {coverage['counts']['road_hypothesis_rows']:,} have a named RoadRegion hypothesis, and {coverage['counts']['between_second_rows']:,} have a bound second anchor for between ({coverage['counts']['between_annotation_second_rows']:,} annotation-derived). Schema omissions and duplicate sample IDs: {coverage['counts']['missing_schema_rows']}/{coverage['counts']['duplicate_sample_ids']}.",
          'Annotation-derived anchor names come from the target object and are an **offline diagnostic oracle**. Parser-derived anchors come only from instruction text. Neither path uses GT position to pick an anchor or road segment.']
    (R/'GATE_F_INPUT_AUDIT.md').write_text('\n'.join(inp)+'\n')
    # Reference frames
    directions=('behind','in front of','left of','right of')
    fr=['# CityNav reference frame report','','Selection uses train_seen statistics and val_seen AUC. The val_unseen column is sealed reporting. FINAL windows use XY motion from the window start to the trajectory end; route uses start-to-end or PCA motion. Full future trajectory is diagnostic evidence about the annotator viewpoint, not an instruction-time feature.','',
        'The anchor frame is geometric footprint orientation and does not identify a facade. Road frame uses a local RoadRegion tangent. Invalid motion windows have no score.']
    table(fr,['Final window','Valid legacy rows','Valid rate','Mean step-heading agreement'],[(n,trajectory['heading_stability'][str(n)]['count'],num(trajectory['heading_stability'][str(n)]['count']/trajectory['samples']),num(trajectory['heading_stability'][str(n)]['mean'])) for n in (3,5,10,20)])
    for phrase in directions:
        fr+=['',f'## {phrase}']
        relevant=[r for r in programs if r['phrase']==phrase and r['context'] and r['source']=='annotation']
        if not relevant:fr+=['No evaluable annotation clauses.'];continue
        relevant.sort(key=lambda r:-r['n_val_seen'])
        for rec in relevant[:3]:
            stem='BACK' if phrase=='behind' else 'FRONT' if phrase=='in front of' else 'LEFT' if phrase=='left of' else 'RIGHT'
            frames=['GLOBAL','START','FINAL3','FINAL5','FINAL10','FINAL20','ROUTE_END','ROUTE_PCA','ROAD','ANCHOR_MAJOR','ANCHOR_MINOR']
            fr+=['',f"Anchor type: {rec['anchor_type']}; train/val seen/val unseen clause counts: {rec['n_train']}/{rec['n_val_seen']}/{rec['n_val_unseen']}. Best: `{rec['best_program']}`; reliability {rec['reliability']:.3f}; unseen sign flip: {rec['unstable_seen_unseen']}."]
            table(fr,['Frame','Val seen AUC','Top-1','MRR','Margin','Val unseen AUC'],[(f,auc(rec,stem+'_'+f,'val_seen'),num(rec['val_seen'][stem+'_'+f]['top1']) if stem+'_'+f in rec['val_seen'] else '—',num(rec['val_seen'][stem+'_'+f]['mrr']) if stem+'_'+f in rec['val_seen'] else '—',num(rec['val_seen'][stem+'_'+f]['margin']) if stem+'_'+f in rec['val_seen'] else '—',auc(rec,stem+'_'+f,'val_unseen')) for f in frames])
    fr+=['','View-dependent relations are not stable when a val_seen-positive frame reverses on val_unseen. Reported unseen results never alter frame selection or probabilities.']
    (R/'CITYNAV_REFERENCE_FRAME_REPORT.md').write_text('\n'.join(fr)+'\n')
    # Road
    rr=['# Road relation canonicalization','','RoadRegion is a named connected component. Candidate geometry uses the nearest local part of an ordered segment-centre polyline; local tangent, normal, signed side, projection and corridor distance are available. The polyline is an approximation, so hairpin roads can have imperfect local ordering. Across-road crossing and opposite side remain undefined where the instruction lacks an independent reference entity on the other side.','',
        f"Legacy diagnostic road graph summary: {road['graphs']} maps; {road['regions']} RoadRegions. The supplemental raw sample has {raw['rows_added']} clauses across on/off/along/across; it uses a separate random map-distractor protocol."]
    for phrase in ('on','off','along','across','across from'):
        rr+=['',f'## {phrase} + road']
        relevant=[r for r in programs if r['phrase']==phrase and r['anchor_type']=='road_region' and r['context']]
        if not relevant:rr+=['No evaluable rows.'];continue
        for rec in relevant:
            rr+=['',f"Source `{rec['source']}`; n train/val seen/val unseen = {rec['n_train']}/{rec['n_val_seen']}/{rec['n_val_unseen']}; selected `{rec['best_program']}`; reliability {rec['reliability']:.3f}; unseen sign flip {rec['unstable_seen_unseen']}."]
            table(rr,['Program','Val seen AUC','Val unseen AUC'],[(p,auc(rec,p,'val_seen'),auc(rec,p,'val_unseen')) for p in sorted(rec['val_seen'])])
    rr+=['','A road name alone cannot define which side an independent landmark occupies. The across controls therefore abstain when a second anchor is missing.']
    (R/'ROAD_RELATION_CANONICALIZATION_REPORT.md').write_text('\n'.join(rr)+'\n')
    # Between
    br=['# Between: true ternary relation','','Only annotation clauses with explicit `between A and B` and both bound anchors enter B0–B4. Annotation names are offline oracle evidence. B3 replaces B with a deterministic unrelated same-kind map entity; B4 uses proximity to A.','']
    for split in ('train_seen','val_seen','val_unseen'):
        br+=['',f'## {split}',f"n = {between['n_explicit_bound_clauses'].get(split,0)}"]
        table(br,['Control','Pairwise AUC','Top-1','MRR'],[(p,num(x['auc']),num(x['top1']),num(x['mrr'])) for p,x in between['program_metrics'].get(split,{}).items() if x])
        table(br,['B2 versus','Clustered wins/losses/ties','Mean AUC difference','One-sided paired p'],[(ast.literal_eval(p)[1],f"{v['wins']}/{v['losses']}/{v['ties']}",num(v['mean_auc_difference']),pnum(v['paired_sign_p_one_sided'])) for p,v in between['B2_paired_comparisons'].items() if p.startswith(f"('{split}',")])
    br+=['','A useful ternary program must beat the pairwise and shuffled-second controls on val_unseen. That condition is checked directly, rather than inferred from its raw AUC.']
    (R/'BETWEEN_TERNARY_RELATION_REPORT.md').write_text('\n'.join(br)+'\n')
    # Lexicon
    lex=['# CityNav spatial lexicon: Gate F evidence','','Probabilities and reliability are fitted from train_seen plus val_seen only. The unseen column is a post-selection stability check. UNKNOWN remains a program hypothesis. Annotation rows are offline oracle diagnostics; raw_parser rows are instruction-only and use easier deterministic map distractors, so their rho is not comparable with legacy hard-negative rows.','']
    def frame_name(program,anchor_type):
        if program.startswith(('BACK_','FRONT_','LEFT_','RIGHT_')):return program.split('_',1)[1]
        if 'ROAD' in program:return 'ROAD_REGION'
        if 'ANCHOR' in program:return 'ANCHOR'
        return 'NONE'
    table(lex,['Phrase','Semantic family','Anchor','Source','n train/seen/unseen','Selected program','Frame','P(selected)','P(UNKNOWN)','rho','Seen AUC','Unseen AUC','Stable'],[(r['phrase'],PHRASES.get(r['phrase'],'UNKNOWN'),r['anchor_type'],r['source'],f"{r['n_train']}/{r['n_val_seen']}/{r['n_val_unseen']}",r['best_program'],frame_name(r['best_program'],r['anchor_type']),num(r['probabilities'].get(r['best_program'])),num(r['probabilities'].get('UNKNOWN')),num(r['reliability']),auc(r,r['best_program'],'val_seen'),auc(r,r['best_program'],'val_unseen'),'no' if r['unstable_seen_unseen'] else 'yes') for r in sorted(programs,key=lambda r:-r['n_val_seen']) if r['context'] and r['n_val_seen']>=5])
    lex+=['','A probability is an exploratory geometric calibration, not DeepSeek confidence. Low counts and unseen sign flips preclude deploying a phrase even when its val_seen AUC exceeds chance.']
    (R/'CITYNAV_SPATIAL_LEXICON.md').write_text('\n'.join(lex)+'\n')
    # Gate
    gate=['# Spatial canonicalization Gate F','','## Decision',f"**{metrics['gate_f']['status']}** under the required conjunction. The measured geometry now includes trajectory windows, named RoadRegions, and explicit second anchors; failure is no longer attributed to those three columns being absent.",'',
          '## Input and parser',f"- Legacy diagnostic samples: 2,290. Extended clause rows: {metrics['source_rows']:,} from {coverage['unique_episodes']:,} unique episodes; evaluable rows: {metrics['evaluated_clause_rows']:,}.",
          f"- Full trajectory: {coverage['counts']['trajectory_rows']:,}/{coverage['counts']['rows']:,}. Named RoadRegion hypothesis: {coverage['counts']['road_hypothesis_rows']:,}/{coverage['counts']['rows']:,}. Explicit bound second-anchor between rows: {coverage['counts']['between_second_rows']:,} (annotation path {coverage['counts']['between_annotation_second_rows']:,}).",
          f"- DeepSeek paired sample: {parser['n']} instructions; valid A/B {parser['valid_a']:.1%}/{parser['valid_b']:.1%}; exact agreement {parser['exact_consistency']:.1%}; target agreement {parser['target_exact']:.1%}; clause Jaccard {parser['clause_jaccard']:.3f}; UNKNOWN {parser['unknown_rate']:.1%}. Its {deepseek['clause_rows_added']} clauses have a separate geometry path ({deepseek['bound_anchor_rows']} bound; {deepseek['canonical_unknown_clauses']} canonical UNKNOWN).",
          '','## Gate conditions']
    table(gate,['Condition','Met'],[(k,str(v)) for k,v in metrics['gate_f']['conditions'].items()])
    gate+=['','## F0–F4 comparison','','F0 executes the existing hard geometry ontology for near and between; unsupported words receive a neutral score. The learned old teacher/DeepSeek pipeline is not reproduced on these exact clause rows. F1 uses one phrase-level program; F2 adds anchor context; F3 marginalizes programs; F4 applies val_seen reliability. Protocols have different negative difficulty and are shown separately.']
    table(gate,['Split','Protocol','Variant','n','Top-1','Top-4','MRR','AUC','Margin'],[(split,protocol,name,m['n'],num(m['top1']),num(m['top4']),num(m['mrr']),num(m['auc']),num(m['margin'])) for key,m in sorted(metrics['variants_by_protocol'].items()) for split,name,protocol in [key.split(':',2)]])
    gate+=['','## Paired variant comparisons on val_unseen','','Each row uses identical candidate sets for both variants; the sign test clusters steps by episode.']
    variant_rows=[]
    for key,v in metrics.get('paired_variants_by_source',{}).items():
        split,name,source,protocol=ast.literal_eval(key)
        if split!='val_unseen':continue
        variant_rows.append((source,protocol,name,v['n_episode_clusters'],
                             f"{v['wins']}/{v['losses']}/{v['ties']}",
                             num(v['mean_auc_difference']),pnum(v['paired_sign_p_one_sided'])))
    table(gate,['Source','Protocol','Variant vs F0','n clusters','W/L/T','AUC Δ','p'],variant_rows)
    context_rows=[]
    for key,v in metrics.get('context_vs_phrase',{}).items():
        split,source,protocol=ast.literal_eval(key)
        if split!='val_unseen':continue
        context_rows.append((source,protocol,v['n_episode_clusters'],
                             f"{v['wins']}/{v['losses']}/{v['ties']}",
                             num(v['mean_auc_difference']),pnum(v['paired_sign_p_one_sided'])))
    table(gate,['Source','Protocol','n clusters','F2 over F1 W/L/T','AUC Δ','p'],context_rows)
    gate+=['','## Controls','','Paired p-values use a one-sided sign test over episode clusters. Shuffled anchors use a deterministic unrelated same-kind map entity. These controls test geometry discrimination but are sensitive to the quality of binding.']
    control_rows=[]
    for key,v in metrics['controls'].items():
        parsed_key=ast.literal_eval(key)
        if len(parsed_key)!=5:continue
        split,phrase,control,source,protocol=parsed_key
        if split!='val_unseen' or phrase not in ('behind','in front of','left of','right of','between','on','off','along','across','next to','near'):continue
        control_rows.append((source,protocol,phrase,control,v['n_episode_clusters'],
                             f"{v['wins']}/{v['losses']}/{v['ties']}",
                             num(v['mean_auc_difference']),pnum(v['paired_sign_p_one_sided'])))
    table(gate,['Source','Protocol','Phrase','Control','n clusters','W/L/T','AUC Δ','p'],control_rows)
    gate+=['','## Failure attribution','','Frame ambiguity and language semantics instability are the dominant failures. The true ternary between program is not recoverable beyond the pairwise control with the current anchor geometry. Anchor resolution remains incomplete; road geometry is available for named regions, while across-road interpretation often lacks an independent second reference.','','## Interpretation','','- Behind/front selected frames flip sign on val_unseen. Left/right remain weak and do not pass their opposite or shuffled-relation controls; no reliable common reference frame is established.','- On-road association has evidence above chance, but the road program family does not establish the full cross-relation gate. Off/along use a separately sampled candidate protocol.','- True ternary between does not beat the old pairwise interpretation on val_unseen. Its shuffled-second control is weaker, which shows the second anchor carries information without establishing the chosen ternary score as best.','- F3 gains over F0 need to be read by covered and uncovered relation separately; unsupported F0 words are neutral by definition.','',
          '## Leakage and split control','','Text-only parser requests contain no target IDs, positions, candidate ranks or answer fields. Graph binding retains all name hypotheses. GT target type and index are read only by the offline evaluator for same-class candidate filtering and metrics. Full trajectory is diagnostic; final approach would not be available at instruction time. Program choice, probabilities and rho use train_seen plus val_seen only.','',
          '## One next step','','Stop the fine-grained relation line and archive these diagnostics, including the limited proximity and named-road association signals.']
    (R/'SPATIAL_CANONICALIZATION_GATE_F_REPORT.md').write_text('\n'.join(gate)+'\n')
if __name__=='__main__':main()
