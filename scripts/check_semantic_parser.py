#!/usr/bin/env python3
"""Paired prompt-order check on a fixed stratified instruction sample."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.spatial_graph.semantic_deepseek import call
from sensaturban_fpv.spatial_graph.deepseek_parser import Cache,require_key
OUT=ROOT/'artifacts/spatial_semantic_canonicalization'
def signature(p):
    return (p.get('target_phrase','').lower().strip(),tuple(sorted((c['raw_phrase'].lower().strip(),c['anchor_phrase'].lower().strip(),c['semantic_family'],c['arity'],c.get('view_dependency')) for c in p['clauses'])))
def main():
    rows=[json.loads(l) for l in (OUT/'parsed_clauses.jsonl').open()]
    by={}
    for r in rows:
        for c in r['clauses']:
            p=c['raw_phrase']
            if p not in by and c['anchor_phrase']:by[p]=r['instruction']
    selected=list(by.items());cache=Cache(OUT/'deepseek_cache');key=require_key()
    result=[]
    for phrase,instruction in selected:
        try:a=call(instruction,key,cache);b=call(instruction,key,cache,True)
        except Exception as exc:a=b=None;error=type(exc).__name__
        else:error=None
        result.append({'phrase':phrase,'instruction':instruction,'prompt_a':a,'prompt_b':b,
                       'valid_a':a is not None,'valid_b':b is not None,'exact':a is not None and b is not None and signature(a)==signature(b),'error':error})
        print(phrase,result[-1]['valid_a'],result[-1]['exact'],flush=True)
    valid=[r for r in result if r['valid_a'] and r['valid_b']]
    summary={'n':len(result),'success_a':sum(r['valid_a'] for r in result)/len(result),
             'success_b':sum(r['valid_b'] for r in result)/len(result),
             'exact_consistency':sum(r['exact'] for r in valid)/len(valid) if valid else None,
             'unknown_rate':sum(any(c['semantic_family']=='UNKNOWN' for c in (r['prompt_a'] or {}).get('clauses',[])) for r in result)/len(result),
             'records':result}
    (OUT/'parser_consistency.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
