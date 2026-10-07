#!/usr/bin/env python3
"""Full split audit. Reads instruction text, never answer fields."""
import json,sys
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.config import load_config
from sensaturban_fpv.spatial_graph.layout_program import parse_layout

def main():
    cfg=load_config();source=Path(cfg['paths']['citynav_dir']);out=ROOT/'artifacts/instruction_defined_layout';out.mkdir(parents=True,exist_ok=True)
    # Labels are consulted only for aggregate audit counts after text parsing.
    object_types={m:{int(i):o['object_type'] for i,o in group.items()} for m,group in json.loads((Path(cfg['paths']['cityrefer_dir'])/'objects.json').read_text()).items()}
    generic={(r['split'],r['episode_index']) for line in (ROOT/'artifacts/spatial_canonicalization_gate_f/parsed_clauses.jsonl').open() if (r:=json.loads(line)).get('clauses')}
    totals=Counter();families=defaultdict(lambda:{'count':0,'split':Counter(),'target_type':Counter(),'contexts':[]});frames=defaultdict(lambda:{'count':0,'contexts':[]});samples=[]
    with (out/'parsed_layout_programs.jsonl').open('w') as parsed_file,(out/'explicit_frame_samples.jsonl').open('w') as frame_file:
        for split in ('train_seen','val_seen','val_unseen'):
            for index,row in enumerate(json.loads((source/f'citynav_{split}.json').read_text())):
                text=(row.get('descriptions') or [''])[0];p=parse_layout(text);layout=bool(p['layout_programs']);frame=p['frame']['is_explicit'];totals['instructions']+=1
                totals['layout']+=layout;totals['frame']+=frame;totals['frame_and_layout']+=layout and frame
                totals[f'{split}_instructions']+=1;totals[f'{split}_layout']+=layout;totals[f'{split}_frame']+=frame;totals[f'{split}_frame_and_layout']+=layout and frame
                group='A' if layout and frame else 'B' if layout else 'C' if not frame and (split,index) in generic else 'OTHER';totals[f'group_{group}']+=1
                if not(layout or frame):continue
                raw_ids=row.get('object_ids') or [];raw=raw_ids[0] if len(raw_ids)==1 else None
                target_id=int(raw.get('object_id',raw.get('id',next(iter(raw.values())))) if isinstance(raw,dict) else raw) if raw is not None else None
                target_type=object_types.get(f"{row['area']}_block_{row['block']}",{}).get(target_id,'unknown')
                if layout:totals[f'layout_target_{target_type}']+=1
                if layout and frame:totals[f'frame_layout_target_{target_type}']+=1
                rec={'split':split,'episode_index':index,'instruction':text,'target_type_offline':target_type,'group':group,**p}
                parsed_file.write(json.dumps(rec)+'\n')
                if frame:frame_file.write(json.dumps(rec)+'\n')
                for q in p['layout_programs']:
                    f=families[q['type']];f['count']+=1;f['split'][split]+=1;f['target_type'][target_type]+=1
                    if len(f['contexts'])<5:f['contexts'].append(text)
                for cue in p['frame']['cues']:
                    f=frames[cue['axis_source']];f['count']+=1
                    if len(f['contexts'])<5:f['contexts'].append(text)
                if len(samples)<100:samples.append(rec)
    audit={'counts':dict(totals),'families':{k:{**v,'split':dict(v['split']),'target_type':dict(v['target_type'])} for k,v in families.items()},'frame_cues':dict(frames),'examples':samples[:20]}
    (out/'layout_language_audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# Explicit layout language audit','',f"CityNav train_seen, val_seen, val_unseen: {totals['instructions']:,} instructions. Text-only regex parsing; target IDs are read only afterward to aggregate actual target types.",'',
           '| Measure | Count | Share |','|---|---:|---:|']
    for key,label in [('layout','Explicit layout program'),('frame','Explicit frame cue'),('frame_and_layout','Frame cue plus layout'),('group_A','Group A: frame plus layout'),('group_B','Group B: layout only'),('group_C','Group C: generic relation only'),('group_OTHER','Other/no matched relation')]:
        lines.append(f"| {label} | {totals[key]:,} | {totals[key]/totals['instructions']:.2%} |")
    lines+=['','## Layout families','','| Family | Count | Actual target type | Splits | Context |','|---|---:|---|---|---|']
    for name,r in sorted(audit['families'].items(),key=lambda x:-x[1]['count']):
        lines.append(f"| {name} | {r['count']} | {r['target_type']} | {r['split']} | {r['contexts'][0][:180].replace('|','/')} |")
    lines+=['','## Frame cues','','| Axis cue | Count | Example |','|---|---:|---|']
    for name,r in sorted(frames.items(),key=lambda x:-x[1]['count']):lines.append(f"| {name} | {r['count']} | {r['contexts'][0][:180].replace('|','/')} |")
    lines+=['','Target types are offline aggregate metadata from CityRefer object IDs; the parser and frame constructor receive instruction text only. Group C contains no layout/frame cue and at least one Gate F parsed generic spatial clause. Other instructions are outside A/B/C. Regex recall and precision are not assumed perfect.']
    (ROOT/'EXPLICIT_LAYOUT_LANGUAGE_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(totals),indent=2))
if __name__=='__main__':main()
