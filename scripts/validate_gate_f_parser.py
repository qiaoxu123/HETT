#!/usr/bin/env python3
"""Controlled high-frequency clause sample, two prompt orders, no answer fields."""
import json,random,sys,os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.spatial_graph.semantic_deepseek import call
from sensaturban_fpv.spatial_graph.deepseek_parser import Cache,require_key
from sensaturban_fpv.anchor_parser import normalise
SOURCE=Path(os.environ.get('GATE_F_SOURCE_ROWS', ROOT.parent/'Achieved/hett-experiments/55-sensaturban-fpv-rendering/artifacts/relation_v2/relation_samples.jsonl'))
OUT=ROOT/'artifacts/spatial_canonicalization_gate_f'
PHRASES=('behind','in front of','between','on','left of','right of','next to','near','across from')

def sig(parsed):
    if not parsed:return None
    return (normalise(parsed['target_phrase']),tuple(sorted((normalise(c['raw_phrase']),normalise(c['anchor_phrase']),c['semantic_family'],c['arity'],c.get('view_dependency')) for c in parsed['clauses'])))

def jaccard(a,b):
    if a is None or b is None:return None
    x=set(a[1]);y=set(b[1]);return len(x&y)/len(x|y) if x|y else 1.

def main():
    pools=defaultdict(dict)
    for line in SOURCE.open():
        row=json.loads(line)
        for clause in row.get('annotation_relations') or []:
            if clause and clause.get('phrase') in PHRASES:
                pools[clause['phrase']][row['instruction']]=None
    rng=random.Random(20261007)
    selected=[]
    for phrase in PHRASES:
        instructions=sorted(pools[phrase]);rng.shuffle(instructions)
        selected += [(phrase,x) for x in instructions[:12]]
    cache=Cache(OUT/'deepseek_cache');key=require_key()
    def work(item):
        phrase,instruction=item
        try:a=call(instruction,key,cache,False);b=call(instruction,key,cache,True);err=None
        except Exception as e:a=b=None;err=type(e).__name__
        sa,sb=sig(a),sig(b)
        return {'phrase':phrase,'instruction':instruction,'prompt_a':a,'prompt_b':b,'valid_a':a is not None,'valid_b':b is not None,
                'exact':sa is not None and sa==sb,'target_exact':sa is not None and sb is not None and sa[0]==sb[0],
                'clause_jaccard':jaccard(sa,sb),
                'raw_phrase_exact':sa is not None and sb is not None and sorted(x[0] for x in sa[1])==sorted(x[0] for x in sb[1]),
                'anchor_phrase_exact':sa is not None and sb is not None and sorted(x[1] for x in sa[1])==sorted(x[1] for x in sb[1]),
                'semantic_family_exact':sa is not None and sb is not None and sorted(x[2] for x in sa[1])==sorted(x[2] for x in sb[1]),
                'arity_exact':sa is not None and sb is not None and sorted(x[3] for x in sa[1])==sorted(x[3] for x in sb[1]),
                'view_dependency_exact':sa is not None and sb is not None and sorted(str(x[4]) for x in sa[1])==sorted(str(x[4]) for x in sb[1]),
                'error':err}
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(work,x) for x in selected]
        for i,f in enumerate(as_completed(futures),1):
            results.append(f.result())
            if i%10==0:print(i,len(selected),flush=True)
    valid=[r for r in results if r['valid_a'] and r['valid_b']]
    summary={'n':len(results),'valid_a':sum(r['valid_a'] for r in results)/len(results),
             'valid_b':sum(r['valid_b'] for r in results)/len(results),
             'exact_consistency':sum(r['exact'] for r in valid)/len(valid) if valid else None,
             'target_exact':sum(r['target_exact'] for r in valid)/len(valid) if valid else None,
             'clause_jaccard':sum(r['clause_jaccard'] for r in valid)/len(valid) if valid else None,
             'field_consistency':{field:sum(r[field+'_exact'] for r in valid)/len(valid) if valid else None
                                  for field in ('raw_phrase','anchor_phrase','semantic_family','arity','view_dependency')},
             'unknown_rate':sum(any(c['semantic_family']=='UNKNOWN' for c in (r['prompt_a'] or {}).get('clauses',[])) for r in results)/len(results),
             'by_phrase':{},'records':results}
    for phrase in PHRASES:
        subset=[r for r in results if r['phrase']==phrase]
        summary['by_phrase'][phrase]={'n':len(subset),'exact_consistency':sum(r['exact'] for r in subset)/len(subset) if subset else None}
    (OUT/'parser_validation.json').write_text(json.dumps(summary,indent=2))
    print({k:v for k,v in summary.items() if k not in ('records','by_phrase')})
if __name__=='__main__':main()
