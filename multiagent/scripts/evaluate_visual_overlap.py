"""Recover every validation footprint and evaluate overlap-only ranking oracles."""
import json
import cv2
import numpy as np
import rasterio
from scipy.stats import spearmanr
from multiagent.visual_goal.diagnosis_data import ROOT,OUT,DATASET,load_split,recover_poses,write_json,file_hash
from multiagent.visual_goal.overlap import crop_transform,valid_mask,coverage,BINS,bin_mask
from multiagent.visual_goal.failure_analysis import summarize


def main():
    (OUT/'cache').mkdir(parents=True,exist_ok=True)
    audit={'numeric_zero_area_tolerance_m2':1e-6,'base_commit':'856e8400246730554ee63e39731761e5fdf72973','maps':{},'splits':{},'coordinate_convention':'world metres, local map-specific affine; OpenCV col,row pixel centres; yaw0 +X up,+Y left', 'old_mask_bug':'old world_to_template exchanges image row/column; preserved original files; new geometry uses exact RGB homography', 'support':'intersection with hull of raster pixel centres; excludes out-of-raster padding, bilinear half-pixel boundary conservative','internal_nodata':'TIFF is elevation, its nodata does not define RGB invalidity; RGB PNG has no alpha; internal unlabelled RGB void cannot be inferred reliably'}
    for split in ('val_seen','val_unseen'):
        q,c=load_split(split);positions=recover_poses(split,q);records=[];geoms=[];masks=[]
        for role,rows in [('q',q),('c',c)]:
            geometries=[];current=None;r=None
            for i,row in enumerate(rows):
                name=row['map_name']
                if name!=current:
                    if r:r.close()
                    r=rasterio.open(ROOT/'data/rgbd'/f'{name}.tif');current=name
                    if name not in audit['maps']:
                        im=cv2.imread(str(ROOT/'data/rgbd'/f'{name}.png'))
                        assert im.shape[:2]==r.shape
                        audit['maps'][name]={'affine':list(r.transform)[:6],'width':r.width,'height':r.height,'crs':str(r.crs),'png_tiff_shape_agree':True,'pixel_size_m':[abs(r.transform.a),abs(r.transform.e)]}
                xy=positions[i] if role=='q' else row['target_xy']
                agl=row['altitude_agl_m'] if role=='q' else 20.
                yaw=row['yaw_rad'] if role=='q' else 0.
                H,full,valid=crop_transform(r.transform,r.width,r.height,xy,agl,yaw)
                mask=valid_mask(H,r.width,r.height)
                geometries.append(valid)
                records.append({'role':role,'index':i,'key':row.get('query_key',row.get('scene_key')),'map_name':name,'xy':list(xy),'agl':agl,'yaw':yaw,'H':H.tolist(),'valid_area_m2':valid.area,'full_area_m2':full.area,'valid_fraction':valid.area/full.area,'valid_polygon':list(valid.exterior.coords) if not valid.is_empty else []})
                masks.append(mask)
            if r:r.close()
            geoms.append(geometries)
        scores={name:np.zeros((len(q),len(c)),np.float32) for name in ('intersection_m2','query_coverage','template_coverage','iou')}
        for i,row in enumerate(q):
            for j,cr in enumerate(c):
                if row['map_name']!=cr['map_name']:continue
                for key,value in coverage(geoms[0][i],geoms[1][j]).items():scores[key][i,j]=value
            if i%2000==0:print('overlap',split,i,len(q),flush=True)
        np.savez_compressed(OUT/'cache'/f'overlap_{split}.npz',**scores)
        np.savez_compressed(OUT/'cache'/f'valid_masks_{split}.npz',q=np.asarray(masks[:len(q)]),c=np.asarray(masks[len(q):]))
        write_json(OUT/'cache'/f'footprints_{split}.json',records)
        pi=np.asarray([next(j for j,x in enumerate(c) if x['scene_key']==row['scene_key']) for row in q])
        result={'queries':len(q),'candidates':len(c),'oracles':{},'overlap_distribution':{}}
        for key in ('query_coverage','template_coverage','iou'):
            pos=scores[key][np.arange(len(q)),pi]
            result['oracles'][key]=summarize(scores[key],q,c,ci=True)
            result['overlap_distribution'][key]={label:int(bin_mask(pos,lo,hi).sum()) for lo,hi,label in BINS}
            corr=spearmanr(pos,[r['distance_to_goal_m'] for r in q]);result[key+'_distance_rho']=float(corr.statistic)
        result['no_intersection_fraction']=float(np.mean(scores['intersection_m2'][np.arange(len(q)),pi]<=1e-8))
        result['valid_fraction_query']={k:float(v) for k,v in zip(('min','median','max'),np.quantile([r['valid_fraction'] for r in records if r['role']=='q'],[0,.5,1]))}
        result['valid_fraction_candidate']={k:float(v) for k,v in zip(('min','median','max'),np.quantile([r['valid_fraction'] for r in records if r['role']=='c'],[0,.5,1]))}
        audit['splits'][split]=result
        write_json(OUT/'overlap_oracle.json',audit)
    audit['source_hashes']={f'{kind}_{split}':file_hash(DATASET/f'{kind}_{split}.jsonl') for kind in ('queries','templates') for split in ('val_seen','val_unseen')}
    write_json(OUT/'overlap_oracle.json',audit)

if __name__=='__main__':main()
