"""Same-gallery global/local RGB, observed/map geometry and fixed-alpha fusion.
Stage outputs cached independently; model inputs exclude all world coordinates.
"""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
import rasterio
from PIL import Image
from shapely.geometry import Polygon,box
from shapely.strtree import STRtree
from multiagent.visual_goal.diagnosis_data import ROOT,DATASET,OUT,load_split,write_json
from multiagent.visual_goal.overlap import world_to_image
from multiagent.visual_goal.local_matching import pool_patches,patch_valid,pair_scores,ransac_correspondences
from multiagent.visual_goal.geometry_matching import observed_edges,small_edges,distance_field,variants
from multiagent.visual_goal.score_fusion import ALPHAS,fuse,select_alpha
from multiagent.visual_goal.failure_analysis import summarize,query_outcomes,paired_comparison

CACHE=OUT/'cache'

class MapOutlines:
    """Offline B2b adapter only. Geometry contains every mapped structure,
    never privileged target/referenced-anchor identity. Same source on both sides.
    """
    def __init__(self):
        self.objects=json.loads((ROOT/'data/cityrefer/objects.json').read_text());self.maps={}
    def render(self,record,valid):
        name=record['map_name']
        if name not in self.maps:
            with rasterio.open(ROOT/'data/rgbd'/f'{name}.tif') as r:
                t=r.transform
                labels=np.zeros((r.height,r.width),np.uint16)
            # Rasterize once in native map pixels; warp each view with the exact
            # RGB homography. Labels only produce boundaries, never descriptors.
            inv=~t
            for j,obj in enumerate(self.objects[name].values()):
                contour=obj.get('contour') or []
                if len(contour)<3:continue
                if obj.get('object_type') not in {'Building','Wall','TrafficRoad','Footpath','Rail','Bridge','Parking'} and not obj.get('name'):continue
                xy=np.asarray(contour,dtype=float)[:,:2]
                col=inv.a*xy[:,0]+inv.b*xy[:,1]+inv.c-.5
                row=inv.d*xy[:,0]+inv.e*xy[:,1]+inv.f-.5
                pts=np.round(np.stack([col,row],-1)).astype(np.int32)
                cv2.fillPoly(labels,[pts],int(j%65534+1))
            self.maps[name]=labels
        im=cv2.warpPerspective(self.maps[name],np.asarray(record['H']),(224,224),flags=cv2.INTER_NEAREST)
        edge=np.zeros_like(im,dtype=bool)
        edge[1:]|=im[1:]!=im[:-1];edge[:,1:]|=im[:,1:]!=im[:,:-1]
        return edge&cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)


def build_geometry(split):
    path=CACHE/f'geometry_inputs_{split}.npz'
    if path.exists():return
    q,c=load_split(split);valid={k:v for k,v in np.load(CACHE/f'valid_masks_{split}.npz').items()};records=json.loads((CACHE/f'footprints_{split}.json').read_text())
    mapper=MapOutlines();out={};offset=0
    for role,rows in [('q',q),('c',c)]:
        observed=[];assisted=[]
        for i,row in enumerate(rows):
            rgb=np.asarray(Image.open(DATASET/row['image_path']).convert('RGB'))
            observed.append(small_edges(observed_edges(rgb,valid[role][i])))
            assisted.append(small_edges(mapper.render(records[offset+i],valid[role][i])))
            if i%500==0:print('geometry inputs',split,role,i,len(rows),flush=True)
        offset+=len(rows)
        out['observed_'+role]=np.asarray(observed);out['map_'+role]=np.asarray(assisted)
    np.savez_compressed(path,**out)


def geometry_scores(qedges,cedges,device='cuda'):
    qe=qedges.reshape(len(qedges),-1);qvalid=qe.sum(1)>=3
    qdt=np.asarray([distance_field(e).ravel() for e in qedges])
    qe=qe/np.maximum(qe.sum(1,keepdims=True),1)
    scores=np.empty((len(qe),len(cedges)),np.float32)
    Q=torch.tensor(qe,device=device);D=torch.tensor(qdt,device=device)
    for start in range(0,len(cedges),16):
        es=[];ds=[]
        for edge in cedges[start:start+16]:
            e,d=variants(edge);es.append(e);ds.append(d)
        E=torch.tensor(np.asarray(es).reshape(-1,1024),device=device);C=torch.tensor(np.asarray(ds).reshape(-1,1024),device=device)
        nv=es[0].shape[0];nc=len(es)
        for i in range(0,len(qe),256):
            loss=(D[i:i+256]@E.T+Q[i:i+256]@C.T)*.5
            loss[:,E.sum(-1)==0]=1.
            scores[i:i+256,start:start+nc]=(1-loss.reshape(-1,nc,nv).min(-1).values).cpu().numpy()
    cvalid=cedges.sum((1,2))>=3
    scores[~qvalid]=0;scores[:,~cvalid]=0
    return scores,qvalid,cvalid


def run_local(split):
    path=CACHE/f'local_scores_{split}.npz'
    if path.exists():return
    q,c=load_split(split);masks=np.load(CACHE/f'valid_masks_{split}.npz')
    qf=pool_patches(np.load(CACHE/f'patch_{split}_q.npy',mmap_mode='r'));cf=pool_patches(np.load(CACHE/f'patch_{split}_c.npy',mmap_mode='r'))
    qv=patch_valid(masks['q']);cv=patch_valid(masks['c'])
    Q=qf.cuda();C=cf.cuda();QV=torch.tensor(qv,device='cuda');CV=torch.tensor(cv,device='cuda')
    out={k:np.empty((len(q),len(c)),np.float32) for k in ('nn','mnn','local_rgb','matches','fit_valid','consistency')}
    with torch.inference_mode():
        for i in range(0,len(q),16):
            for j in range(0,len(c),64):
                r=pair_scores(Q[i:i+16,None],C[None,j:j+64],QV[i:i+16,None],CV[None,j:j+64])
                for key,x in r.items():out[key][i:i+16,j:j+64]=x.cpu().numpy()
            if i%640==0:print('local matching',split,i,len(q),flush=True)
    out['query_valid']=qv.sum(1)>=4;out['candidate_valid']=cv.sum(1)>=4
    np.savez_compressed(path,**out)
    # Fixed evenly spaced 200 queries, positive + hardest local same-map negative.
    # This is only a correspondence failure audit, not a selected retrieval pool.
    o,_,pi=query_outcomes(out['local_rgb'],q,c);rs=[]
    for i in np.linspace(0,len(q)-1,200,dtype=int):
        for label,j in [('positive',pi[i]),('hard_negative',o['negative_index'][i])]:
            r=ransac_correspondences(qf[i].numpy(),cf[j].numpy(),qv[i],cv[j]);r.update(query=int(i),candidate=int(j),role=label);rs.append(r)
    write_json(OUT/f'ransac_audit_{split}.json',rs)


def run_geometry(split):
    build_geometry(split);g=np.load(CACHE/f'geometry_inputs_{split}.npz')
    for name in ('observed','map'):
        path=CACHE/f'{name}_scores_{split}.npz'
        if path.exists():continue
        scores,qv,cv=geometry_scores(g[name+'_q'],g[name+'_c'])
        np.savez_compressed(path,scores=scores,query_valid=qv,candidate_valid=cv)
        print('geometry matching complete',name,split,flush=True)


def load_scores(split):
    q,c=load_split(split)
    global_scores=np.load(CACHE/f'global_{split}_q.npy')@np.load(CACHE/f'global_{split}_c.npy').T
    local=np.load(CACHE/f'local_scores_{split}.npz')
    obs=np.load(CACHE/f'observed_scores_{split}.npz');mp=np.load(CACHE/f'map_scores_{split}.npz')
    scores={'global_rgb':global_scores,'local_nn':local['nn'],'local_mnn':local['mnn'],'local_rgb':local['local_rgb'],'geometry_observed':obs['scores'],'geometry_map_selfmatch':mp['scores']}
    # Candidate pool NEVER narrowed for missing features: zero-score abstention.
    # Supplement fair complete-case queries for which every original same-map
    # candidate and query has >=3 structural pixels / >=4 valid patch cells.
    cv=local['candidate_valid']&obs['candidate_valid']&mp['candidate_valid']
    complete=local['query_valid']&obs['query_valid']&mp['query_valid']
    for i,row in enumerate(q):complete[i]&=all(cv[j] for j,x in enumerate(c) if x['map_name']==row['map_name'])
    return q,c,scores,complete,local,obs,mp


def evaluate():
    result={'protocol':{'gallery':'all original 40m scene candidates; context duplicates have identical RGB; same-map negatives unchanged', 'tie_policy':'fractional expected Recall at tied ranks; strict hard-negative win with tie reported separately', 'bootstrap':'1000 resamples of episode clusters, paired sign-flip 4999 permutations; pooled AUROC CI 300 cluster resamples with 1024-bin score quantization', 'fusion':'per-query percentile calibration; alpha grid 0,.25,.5,.75,1 selected on val_seen hard accuracy; val_seen is development not independent final evidence'},'splits':{}}
    alpha=None
    for split in ('val_seen','val_unseen'):
        q,c,s,complete,local,obs,mp=load_scores(split)
        grid={str(a):summarize(fuse(s['local_rgb'],s['geometry_observed'],a),q,c) for a in ALPHAS}
        if split=='val_seen':alpha=select_alpha(grid,split);result['alpha_selected_on_val_seen']=alpha
        s['fusion']=fuse(s['local_rgb'],s['geometry_observed'],alpha)
        split_result={'methods':{},'alpha_grid':grid,'common_valid_n':int(complete.sum()),'common_valid':{},'coverage':{},'paired':{}}
        for name,scores in s.items():
            split_result['methods'][name]=summarize(scores,q,c,ci=True)
            split_result['common_valid'][name]=summarize(scores,q,c,complete)
            print('metrics',split,name,split_result['methods'][name]['R@1'],flush=True)
        for name,data in [('local',local),('observed',obs),('map',mp)]:
            split_result['coverage'][name]={'queries':int(data['query_valid'].sum()),'candidates':int(data['candidate_valid'].sum()),'total_queries':len(q),'total_candidates':len(c)}
        pi=np.asarray([next(j for j,x in enumerate(c) if x['scene_key']==r['scene_key']) for r in q]);ii=np.arange(len(q))
        split_result['coverage']['local']['positive_insufficient_or_degenerate_fit_rate']=float(np.mean(local['fit_valid'][ii,pi]==0))
        split_result['coverage']['local']['all_pair_insufficient_or_degenerate_fit_rate']=float(np.mean(local['fit_valid']==0))
        ep=[r['episode_key'] for r in q];outcomes={k:query_outcomes(v,q,c)[0] for k,v in s.items()}
        for a,b in [('local_rgb','global_rgb'),('geometry_observed','global_rgb'),('geometry_map_selfmatch','global_rgb'),('fusion','local_rgb'),('fusion','geometry_observed')]:
            split_result['paired'][a+'_minus_'+b]={k:paired_comparison(outcomes[a][k],outcomes[b][k],ep) for k in ('R@1','hard_accuracy')}
        result['splits'][split]=split_result;write_json(OUT/'geometry_vs_rgb.json',result)
        np.save(CACHE/f'fusion_scores_{split}.npy',s['fusion'])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['geometry','local','metrics'],required=True);a=p.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(4)
    if a.stage=='metrics':evaluate()
    else:
        for split in ('val_seen','val_unseen'):
            (run_geometry if a.stage=='geometry' else run_local)(split)

if __name__=='__main__':main()
