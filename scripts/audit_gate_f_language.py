#!/usr/bin/env python3
"""Audit all three CityNav splits without reading answer fields for parsing."""
from __future__ import annotations
import json, sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.spatial_graph.semantic_canonicalizer import REGEX,anchor_type,parse_clauses
from sensaturban_fpv.config import load_config

def main():
    cfg=load_config(); source=Path(cfg['paths']['citynav_dir']);out=ROOT/'artifacts/spatial_canonicalization_gate_f';out.mkdir(parents=True,exist_ok=True)
    counts=Counter(); raw_counts=Counter(); records=defaultdict(lambda:{'anchor_types':Counter(),'target_types':Counter(),'contexts':Counter(),'multi_anchor':0})
    total=0;with_rel=0;parse_path=out/'parsed_clauses.jsonl'
    with parse_path.open('w') as f:
        for split in ('train_seen','val_seen','val_unseen'):
            for idx,row in enumerate(json.loads((source/f'citynav_{split}.json').read_text())):
                total+=1;instruction=(row.get('descriptions') or [''])[0]; parsed=parse_clauses(instruction)
                raw_counts.update(m.group().lower() for m in REGEX.finditer(instruction))
                if parsed['clauses']:with_rel+=1
                f.write(json.dumps({'split':split,'episode_index':idx,'instruction':instruction,**parsed})+'\n')
                for clause in parsed['clauses']:
                    p=clause['raw_phrase'];kind=anchor_type(clause['anchor_phrase']);counts[(split,p)]+=1
                    rec=records[p];rec['anchor_types'][kind]+=1;rec['target_types'][anchor_type(parsed['target_phrase'])]+=1
                    rec['contexts'][instruction[:240]]+=1
                    rec['multi_anchor']+=int(len(parsed['clauses'])>1 or p=='between')
    audit={'episodes':total,'episodes_with_relation':with_rel,'phrase_total':sum(counts.values()),
           'splits':{s:dict((p,n) for (sp,p),n in counts.items() if sp==s) for s in ('train_seen','val_seen','val_unseen')},
           'raw_lexical_phrase_total':sum(raw_counts.values()),'raw_lexical_counts':dict(raw_counts),
           'validated_spatial_clause_total':sum(counts.values()),
           'phrases':{p:{'count':sum(n for (sp,q),n in counts.items() if p==q),
                         'anchor_types':dict(r['anchor_types']),'target_types':dict(r['target_types']),
                         'multi_anchor_rate':r['multi_anchor']/sum(n for (sp,q),n in counts.items() if p==q),
                         'road_anchor_rate':r['anchor_types']['road']/sum(n for (sp,q),n in counts.items() if p==q),
                         'building_anchor_rate':r['anchor_types']['building']/sum(n for (sp,q),n in counts.items() if p==q),
                         'contexts':[x for x,_ in r['contexts'].most_common(5)]} for p,r in records.items()}}
    (out/'language_audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# Gate F spatial language audit','',f"Source: all {total:,} train_seen, val_seen, val_unseen episodes; {with_rel:,} contain a matched phrase.",f"Validated clause occurrences: {audit['phrase_total']:,}; raw lexical hits: {audit['raw_lexical_phrase_total']:,}. Counts include overlapping clauses.",'','| Phrase | Count | Anchor types | Target types | Multi anchor | Road anchor | Building anchor | Context |','|---|---:|---|---|---:|---:|---:|---|']
    for p,r in sorted(audit['phrases'].items(),key=lambda x:-x[1]['count']):
        context=r['contexts'][0].replace('|','/') if r['contexts'] else ''
        lines.append(f"| {p} | {r['count']} | {r['anchor_types']} | {r['target_types']} | {r['multi_anchor_rate']:.2f} | {r['road_anchor_rate']:.2f} | {r['building_anchor_rate']:.2f} | {context} |")
    lines+=['','Anchor and target types are lexical estimates from instruction text. The audit never reads target IDs or positions. Raw lexical hits can be false positives, especially short prepositions. The validated clause count rejects bare “in” and requires a typed anchor after bare “on” or “off”; it remains a conservative rule-based estimate, not a gold annotation.']
    (ROOT/'GATE_F_SPATIAL_LANGUAGE_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(total,audit['phrase_total'],len(audit['phrases']))
if __name__=='__main__':main()
