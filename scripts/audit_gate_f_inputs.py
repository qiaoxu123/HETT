#!/usr/bin/env python3
"""Verify sampled-row joins against raw CityNav and CityRefer, with values checked."""
import json,sys,os
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.config import load_config,load_landmarks
from sensaturban_fpv import citynav
SOURCE=Path(os.environ.get('GATE_F_SOURCE_ROWS', ROOT.parent/'Achieved/hett-experiments/55-sensaturban-fpv-rendering/artifacts/relation_v2/relation_samples.jsonl'))

def main():
    cfg=load_config();objects=load_landmarks(cfg);rows=[json.loads(x) for x in SOURCE.open()]
    needed={s:{r['episode_index'] for r in rows if r['split']==s} for s in ('train_seen','val_seen','val_unseen')}
    episodes={}
    for split,indices in needed.items():
        eps=citynav.load_split(Path(cfg['paths']['citynav_dir']),split)
        episodes.update({(split,i):eps[i] for i in indices})
    counts=Counter();examples={}
    for r in rows:
        ep=episodes.get((r['split'],r['episode_index']))
        if ep is None:counts['episode_missing']+=1;continue
        counts['episode_join']+=1
        checks={'instruction':ep.description==r['instruction'],'map':ep.map_name==r['map'],
                'target_id':len(ep.object_ids)==1 and int(ep.object_ids[0])==int(r['target_id']),
                'trajectory':ep.trajectory.ndim==2 and ep.trajectory.shape[1]==6 and len(ep.trajectory)>1,
                'sample_step':0<=r['step']<len(ep.trajectory) and np.allclose(ep.trajectory[r['step'],:2],r['uav_xy']),
                'annotation_id':bool(ep.ann_ids) and int(ep.object_ids[0]) in objects.get(ep.map_name,{}) and ep.ann_ids[0]<len(objects[ep.map_name][int(ep.object_ids[0])].processed_descriptions)}
        for name,ok in checks.items():
            counts[name+('_ok' if ok else '_bad')]+=1
            if not ok:examples.setdefault(name,{'split':r['split'],'episode_index':r['episode_index']})
        if checks['annotation_id']:
            obj=objects[ep.map_name][int(ep.object_ids[0])]
            ann=obj.processed_descriptions[ep.ann_ids[0]]
            counts['target_position_ok']+=int(np.isfinite(np.asarray(obj.position)).all())
            counts['target_footprint_ok']+=int(len(obj.contour)>=3)
            counts['description_landmarks_present']+=int(bool(ann.landmarks))
            counts['two_landmarks_present']+=int(len(ann.landmarks)>=2)
            counts['annotation_text_match']+=int(ep.ann_ids[0]<len(obj.descriptions) and obj.descriptions[ep.ann_ids[0]].strip()==ep.description.strip())
            counts['annotation_whitespace_normalized_match']+=int(ep.ann_ids[0]<len(obj.descriptions) and ' '.join(obj.descriptions[ep.ann_ids[0]].split())==' '.join(ep.description.split()))
    out=ROOT/'artifacts/spatial_canonicalization_gate_f';out.mkdir(parents=True,exist_ok=True)
    (out/'input_audit.json').write_text(json.dumps({'samples':len(rows),'counts':dict(counts),'first_mismatch':examples},indent=2))
    lines=['# Gate F input audit','',f"Inspected {len(rows):,} existing diagnostic rows by joining `split + episode_index` to raw CityNav episodes and `object_ids[0] + ann_ids[0]` to CityRefer objects/processed descriptions.",'',
           '| Checked source value | Valid rows |','|---|---:|']
    labels={'episode_join':'Episode index resolves','instruction_ok':'Instruction exact match','map_ok':'Map/block exact match','target_id_ok':'Target ID exact match (offline only)',
            'trajectory_ok':'Trajectory has >1 steps, six columns','sample_step_ok':'Saved UAV XY matches trajectory step','annotation_id_ok':'Annotation ID resolves','annotation_text_match':'Annotation description exact match','annotation_whitespace_normalized_match':'Annotation description whitespace-normalized match',
            'target_position_ok':'Finite target object position','target_footprint_ok':'Target contour with >=3 points','description_landmarks_present':'Processed description has landmarks','two_landmarks_present':'Processed description has >=2 landmarks'}
    for k,label in labels.items():lines.append(f"| {label} | {counts[k]:,} |")
    lines+=['','CityNav trajectory columns are verified from `citynav._load_split_episodes`: `(x,y,z,dx,dy,dz)`. The last three are recorded look directions; motion headings are computed separately from successive XYZ positions.','',
            'RoadRegion is built from CityRefer TrafficRoad objects by name and connected component in `spatial_graph.road_region`. Named road binding must retain every matching region. Building contours come from the same CityRefer object loader.','',
            'A second anchor is **not** implied by two landmarks alone. It must appear in a `between A and B` clause and both names must bind. Target positions and IDs are read only in this audit and offline evaluation.']
    (ROOT/'GATE_F_INPUT_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(len(rows),dict(counts))
if __name__=='__main__':main()
