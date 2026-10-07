#!/usr/bin/env python3
"""Conservative parking-layout diagnostic; scope and frames use text/map only."""
import json,sys,re,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from shapely.geometry import Point,Polygon
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sensaturban_fpv.config import load_config,load_landmarks
from sensaturban_fpv.spatial_graph.layout_program import (parking_layout,score_order,footprint_axes,oriented_frame,road_s,road_sequence_score)

OUT=ROOT/'artifacts/instruction_defined_layout'
def words(s):return set(re.findall(r'[a-z]{3,}',s.lower()))-{'the','with','from','road','parking','building','house','cars','being','viewed','aerially'}
def anchor_in_text(text,objects):
    normalized=text.lower();hits=[]
    for obj in objects:
        name=obj.name.lower().strip()
        if len(name)>4 and name in normalized:hits.append(obj)
    return sorted(hits,key=lambda o:(-len(o.name),o.id))
def select_scope(text,objects):
    """No target supplied. Unique named reference determines one parking area."""
    cars=[o for o in objects if o.object_type=='Car'];lots=[o for o in objects if o.object_type=='Parking']
    anchors=[o for o in anchor_in_text(text,objects) if o.object_type not in ('Car','TrafficRoad','Parking')]
    if not lots:return None,'no_parking_region',anchors
    if len(lots)==1:lot=lots[0]
    elif anchors:
        a=anchors[0];dist=np.array([np.linalg.norm(np.asarray(o.position[:2])-np.asarray(a.position[:2])) for o in lots]);
        if len(dist)>1 and np.partition(dist,1)[1]-dist.min()<10:return None,'ambiguous_parking_region',anchors
        lot=lots[int(np.argmin(dist))]
    else:return None,'ambiguous_parking_region',anchors
    if len(lot.contour)<3:return None,'invalid_parking_polygon',anchors
    poly=Polygon(lot.contour).buffer(2)
    selected=[o for o in cars if poly.covers(Point(o.position[:2]))]
    if len(selected)<3:return None,'too_few_cars',anchors
    return (lot,selected),'ok',anchors
def cue_frame(cues,objects,scope_center):
    """Only an anchor position relative to the scope can resolve frame sign."""
    for cue in cues:
        direction=cue['direction'];reference=words(cue['reference_phrase'])
        if direction=='unknown':continue
        candidates=[]
        for obj in objects:
            name=words(obj.name)
            if not name or not reference:continue
            overlap=len(name & reference)/len(name)
            if overlap>=.6:candidates.append((overlap,len(name),obj))
        if not candidates:continue
        candidates.sort(key=lambda x:(-x[0],-x[1],x[2].id))
        tied=[c[2] for c in candidates if c[:2]==candidates[0][:2]]
        if len(tied)>1:
            # A named road is stored as many TrafficRoad pieces. Choose the
            # piece nearest the text-selected parking scope, never the GT.
            if not all(o.object_type=='TrafficRoad' and o.name==tied[0].name for o in tied):continue
            obj=min(tied,key=lambda o:np.linalg.norm(np.asarray(o.position[:2])-scope_center))
        else:obj=candidates[0][2]
        if cue['axis_source'] in ('building_long_axis','building_short_axis'):
            axes=footprint_axes(obj.contour)
            if axes is None:continue
            # A footprint axis is unoriented. "long side to the right" does not
            # identify which physical *end* is right without a second cue.
            continue
        v=np.asarray(obj.position[:2],float)-scope_center
        if np.linalg.norm(v)<3:continue
        return oriented_frame(v,direction),obj.id,'resolved_by_reference_position'
    return None,None,'no_oriented_reference'
def score_programs(programs,centers,frame,soft=True,row_start=None,strict=False):
    layout=parking_layout(centers,frame);scores=np.ones(len(centers))
    active=[]
    for p in programs:
        if p['scope']=='road_sequence':continue
        # Baselines assume bottom-first; L3 requires an instruction-defined
        # start. Both choices are recorded rather than selected from GT.
        if p['type']=='ROW_INDEX' and strict and row_start is None:continue
        q={**p,'start':row_start or 'bottom'} if p['type']=='ROW_INDEX' else p
        s=score_order(q,layout['local'],layout['rows'],layout['columns'],1. if soft else .08)
        if np.any(s):scores*=s;active.append(p['type'])
    return (scores if active else None),layout,active
def metric(scores,target):
    if scores is None:return None
    scores=np.asarray(scores,float)
    better=int(np.sum(scores>scores[target]+1e-10));ties=int(np.sum(np.isclose(scores,scores[target],atol=1e-10)))
    rank=better+(ties+1)/2
    other=np.delete(scores,target);auc=float(np.mean(scores[target]>other)+.5*np.mean(scores[target]==other))
    return {'top1':(1/ties if better==0 else 0.),'top4':max(0.,min(4-better,ties)/ties),
            'mrr':float(np.mean(1/np.arange(better+1,better+ties+1))),'auc':auc,'rank':rank,'margin':float(scores[target]-np.max(other))}
def summarize(rows):
    out={}
    for key in ('L0','L1','L2','L3','L4','L5','L6','L7','L8'):
        ms=[r['metrics'][key] for r in rows if r['metrics'].get(key)]
        out[key]={'n':len(ms),**({k:float(np.mean([m[k] for m in ms])) for k in ('top1','top4','mrr','auc','rank','margin')} if ms else {})}
    return out
def main():
    cfg=load_config();objects_by_map=load_landmarks(cfg);parsed=[json.loads(s) for s in (OUT/'parsed_layout_programs.jsonl').open()]
    needed={(p['split'],p['episode_index']) for p in parsed if p['layout_programs']}
    episodes={}
    for split in ('train_seen','val_seen','val_unseen'):
        source=json.loads((Path(cfg['paths']['citynav_dir'])/f'citynav_{split}.json').read_text())
        for i in (j for s,j in needed if s==split):episodes[(split,i)]=source[i]
        del source
    evaluated=[];status=Counter();parking_artifacts=[];frames=[];counterfactual=[]
    for p in parsed:
        if not p['layout_programs']:continue
        ep=episodes[(p['split'],p['episode_index'])];map_name=f"{ep['area']}_block_{ep['block']}";objects=list(objects_by_map.get(map_name,{}).values())
        scope,reason,anchors=select_scope(p['instruction'],objects)
        if scope is None:status[reason]+=1;continue
        lot,candidates=scope;candidate_ids=[o.id for o in candidates]
        # Answer is used only here and below for offline evaluation.
        target_ids=ep.get('object_ids') or []
        if len(target_ids)!=1:status['unsupported_target_record']+=1;continue
        raw_id=target_ids[0]
        target_id=int(raw_id.get('object_id',raw_id.get('id',next(iter(raw_id.values())))) if isinstance(raw_id,dict) else raw_id)
        target_obj=objects_by_map[map_name].get(target_id)
        if target_obj is None or target_obj.object_type!='Car':status['non_car_target']+=1;continue
        if target_id not in candidate_ids:status['scope_miss']+=1;continue
        target=candidate_ids.index(target_id);centers=np.array([o.position[:2] for o in candidates],float)
        center=np.asarray(lot.position[:2],float);explicit,cue_ref,cue_status=cue_frame(p['frame']['cues'],objects,center)
        global_frame=np.eye(2)
        anchor=anchors[0] if anchors else None
        prior=np.exp(-np.linalg.norm(centers-np.asarray(anchor.position[:2]),axis=1)/30.) if anchor else np.ones(len(centers))
        metrics={'L0':metric(prior,target)};score_arrays={'L0':prior};hard_row_auc=None
        row_start=next((cue['direction'] for cue in p['frame']['cues'] if cue['axis_source']=='row_axis'),None)
        for key,frame in [('L1',global_frame),('L2',None),('L3',explicit)]:
            scores,layout,active=score_programs(p['layout_programs'],centers,frame,soft=key!='L3',row_start=row_start,strict=key=='L3') if key!='L3' or explicit is not None else (None,None,[])
            metrics[key]=metric(scores,target);score_arrays[key]=scores
            if key=='L3' and scores is not None:
                neighbors=np.flatnonzero(layout['rows']==layout['rows'][target]);neighbors=neighbors[neighbors!=target]
                if len(neighbors):hard_row_auc=float(np.mean(scores[target]>scores[neighbors])+.5*np.mean(scores[target]==scores[neighbors]))
            if key=='L2' and layout is not None and len(parking_artifacts)<200:
                parking_artifacts.append({'split':p['split'],'episode_index':p['episode_index'],'map_id':map_name,'lot_id':lot.id,'candidate_count':len(candidates),'frame':layout['frame'].tolist(),'row_count':len(set(layout['rows'])),'column_count':len(set(layout['columns'])),'active':active})
        base=score_arrays['L3']
        metrics['L4']=metric(.5*base+.5*prior,target) if base is not None else None
        soft_scores,_,_=score_programs(p['layout_programs'],centers,explicit,soft=True,row_start=row_start,strict=True) if explicit is not None else (None,None,None)
        metrics['L5']=metric(soft_scores,target)
        modified=[{**q,'index':q['index']+1 if isinstance(q['index'],int) and q['index']>0 else q['index']} for q in p['layout_programs']]
        shuffle,_,_=score_programs(modified,centers,explicit,row_start=row_start,strict=True) if explicit is not None else (None,None,None)
        metrics['L6']=metric(shuffle,target)
        flipped,_,_=score_programs(p['layout_programs'],centers,-explicit,row_start=row_start,strict=True) if explicit is not None else (None,None,None)
        metrics['L7']=metric(flipped,target)
        other=next((a for a in objects if a.id!=cue_ref and a.object_type=='Building'),None)
        otherframe=oriented_frame(np.asarray(other.position[:2])-center,'right') if other is not None and explicit is not None else None
        refshuffle,_,_=score_programs(p['layout_programs'],centers,otherframe,row_start=row_start,strict=True) if otherframe is not None else (None,None,None)
        metrics['L8']=metric(refshuffle,target)
        row={'split':p['split'],'episode_index':p['episode_index'],'group':p['group'],'map_id':map_name,'lot_id':lot.id,'candidate_ids':candidate_ids,'target_id_offline':target_id,'target_index_offline':target,'family':[x['type'] for x in p['layout_programs']],'cue_status':cue_status,'same_row_auc':hard_row_auc,'metrics':metrics}
        evaluated.append(row);status['evaluated']+=1
        if explicit is not None:frames.append({'split':p['split'],'episode_index':p['episode_index'],'frame':explicit.tolist(),'reference_id':cue_ref,'source':cue_status})
        if base is not None:counterfactual.append({'split':p['split'],'episode_index':p['episode_index'],'index_changes':not np.allclose(base,shuffle),'frame_changes':not np.allclose(base,flipped),'reference_changes':refshuffle is not None and not np.allclose(base,refshuffle)})
    (OUT/'layout_scores.json').write_text(json.dumps(evaluated,indent=2));(OUT/'local_frames.json').write_text(json.dumps(frames,indent=2))
    (OUT/'parking_layouts.json').write_text(json.dumps(parking_artifacts,indent=2));(OUT/'layout_counterfactuals.json').write_text(json.dumps(counterfactual,indent=2))
    summary={'status':dict(status),'splits':{s:summarize([r for r in evaluated if r['split']==s]) for s in ('train_seen','val_seen','val_unseen')},'groups':{g:summarize([r for r in evaluated if r['group']==g]) for g in ('A','B','C')}}
    for s in ('val_seen','val_unseen'):
        summary.setdefault('families',{})[s]={k:summarize([r for r in evaluated if r['split']==s and k in r['family']]) for k in sorted({f for r in evaluated for f in r['family']})}
    tests={}
    for label,a,b in [('layout_vs_index','L3','L6'),('frame_vs_shuffle','L3','L7'),('frame_vs_global','L3','L1')]:
        paired=[r for r in evaluated if r['split']=='val_unseen' and r['metrics'].get(a) and r['metrics'].get(b)]
        diff=[r['metrics'][a]['auc']-r['metrics'][b]['auc'] for r in paired];wins=sum(x>1e-9 for x in diff);losses=sum(x<-1e-9 for x in diff)
        tests[label]={'n':len(diff),'wins':wins,'losses':losses,'ties':len(diff)-wins-losses,'mean_auc_delta':float(np.mean(diff)) if diff else None,'one_sided_p':float(binomtest(wins,wins+losses,.5,alternative='greater').pvalue) if wins+losses else None}
    summary['paired_tests']=tests
    # The preregistered conjunction needs at least two families and reliable
    # unseen transfer. Empty cells cannot be promoted to success.
    summary['gate_g']='FAIL';summary['reason']='Insufficient resolved explicit-frame layout evaluation and no significant unseen paired evidence.'
    (OUT/'gate_g_metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'status':dict(status),'val_seen':summary['splits']['val_seen'],'val_unseen':summary['splits']['val_unseen'],'paired':tests},indent=2))
if __name__=='__main__':main()
