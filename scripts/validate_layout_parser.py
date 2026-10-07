#!/usr/bin/env python3
"""Controlled two-prompt layout parsing; only instruction text reaches DeepSeek."""
import json,random,sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.spatial_graph.deepseek_parser import Cache,require_key
from sensaturban_fpv.spatial_graph.layout_deepseek import call
OUT=ROOT/'artifacts/instruction_defined_layout'
FAMILIES=('ROW_INDEX','NTH_FROM_LEFT','NTH_FROM_RIGHT','NTH_FROM_TOP','NTH_FROM_BOTTOM','COLUMN_INDEX','NTH_FROM_CORNER','HALFWAY_ALONG')
def signature(p):
    if p is None:return None
    f=p['frame'];ops=tuple(sorted((q['type'],q['index'],q['scope']) for q in p['layout_programs']))
    return (f['is_explicit'],f.get('reference_phrase','').strip().lower(),f['axis_source'],
            tuple(sorted((c['entity'].strip().lower(),c['direction']) for c in f['orientation_constraints'])),ops)
def main():
    pools=defaultdict(dict)
    for line in (OUT/'parsed_layout_programs.jsonl').open():
        r=json.loads(line)
        for q in r['layout_programs']:
            if q['type'] in FAMILIES:pools[q['type']][r['instruction']]=None
    rng=random.Random(20261007);selected=[];seen=set()
    for family in FAMILIES:
        items=sorted(pools[family]);rng.shuffle(items)
        for instruction in items[:20]:
            if instruction not in seen:selected.append((family,instruction));seen.add(instruction)
    cache=Cache(OUT/'deepseek_cache');key=require_key()
    def work(item):
        family,instruction=item
        try:a=call(instruction,key,cache,False);b=call(instruction,key,cache,True);error=None
        except Exception as e:a=b=None;error=type(e).__name__
        sa,sb=signature(a),signature(b);x=set(sa[-1]) if sa else set();y=set(sb[-1]) if sb else set()
        return {'family':family,'instruction':instruction,'prompt_a':a,'prompt_b':b,'valid_a':a is not None,'valid_b':b is not None,
                'exact':sa is not None and sa==sb,'frame_exact':sa is not None and sb is not None and sa[:4]==sb[:4],
                'reference_exact':sa is not None and sb is not None and sa[1]==sb[1],
                'operator_jaccard':len(x&y)/len(x|y) if sa is not None and sb is not None and x|y else 1. if sa is not None and sb is not None else None,
                'ordinal_exact':sa is not None and sb is not None and sorted((v[0],v[1]) for v in x)==sorted((v[0],v[1]) for v in y),'error':error}
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(work,x) for x in selected]
        for i,f in enumerate(as_completed(futures),1):
            results.append(f.result())
            if i%20==0:print(i,len(selected),flush=True)
    valid=[r for r in results if r['valid_a'] and r['valid_b']]
    summary={'n':len(results),'valid_both':len(valid),'exact':sum(r['exact'] for r in valid)/len(valid) if valid else None,
             'frame_exact':sum(r['frame_exact'] for r in valid)/len(valid) if valid else None,
             'reference_exact':sum(r['reference_exact'] for r in valid)/len(valid) if valid else None,
             'operator_jaccard':sum(r['operator_jaccard'] for r in valid)/len(valid) if valid else None,
             'ordinal_exact':sum(r['ordinal_exact'] for r in valid)/len(valid) if valid else None,
             'by_family':{name:{'n':sum(r['family']==name for r in results),'valid_both':sum(r['family']==name for r in valid)} for name in FAMILIES},'records':results}
    (OUT/'parser_validation.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k not in ('records','by_family')})
if __name__=='__main__':main()
