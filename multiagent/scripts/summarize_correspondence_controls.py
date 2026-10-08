"""Full-pool optional correspondence/geometry controls and complete-case audit."""
import json
import numpy as np
from multiagent.visual_goal.diagnosis_data import OUT,write_json
from multiagent.visual_goal.failure_analysis import summarize,query_outcomes,paired_comparison,clustered_mean_ci
from multiagent.visual_goal.score_fusion import fuse,select_alpha,ALPHAS
from multiagent.scripts.evaluate_geometry_vs_rgb import load_scores


def main():
    result={'protocol':'Additional fixed controls; none selected or parameter-tuned using val_unseen. Original DINO fusion retained.', 'splits':{}}
    alpha=None
    for split in ('val_seen','val_unseen'):
        q,c,s,complete,local,obs,mp=load_scores(split);cache=OUT/'cache'
        ra=np.load(cache/f'ransac_scores_{split}.npz');sift=np.load(cache/f'sift_scores_{split}.npz')
        s['dino_ransac']=ra['scores'];s['sift_ransac']=sift['scores'];s['fusion']=np.load(cache/f'fusion_scores_{split}.npy')
        for kind in ('observed','map'):
            g=np.load(cache/f'geometry_controls_{kind}_{split}.npz')
            for name in g.files:s[kind+'_'+name]=g[name]
        grid={str(a):summarize(fuse(s['sift_ransac'],s['geometry_observed'],a),q,c) for a in ALPHAS}
        if split=='val_seen':alpha=select_alpha(grid,split);result['sift_fusion_alpha_selected_on_val_seen']=alpha
        s['sift_fusion']=fuse(s['sift_ransac'],s['geometry_observed'],alpha)
        ov=np.load(cache/f'overlap_{split}.npz');pi=np.asarray([next(j for j,x in enumerate(c) if x['scene_key']==r['scene_key']) for r in q]);ix=np.arange(len(q))
        high=(ov['query_coverage'][ix,pi]>.5)&(ov['template_coverage'][ix,pi]>.5);ep=np.asarray([r['episode_key'] for r in q])
        baseline=query_outcomes(s['global_rgb'],q,c)[0]
        extra=['dino_ransac','sift_ransac','sift_fusion','observed_contour_iou','observed_structural_layout','map_contour_iou','map_structural_layout']
        x={'methods':{},'high_both':{},'paired':{},'paired_high_both':{},'fusion_grid':grid,'availability':{}}
        for name in extra:
            x['methods'][name]=summarize(s[name],q,c,ci=True);x['high_both'][name]=summarize(s[name],q,c,high,ci=True)
            o=query_outcomes(s[name],q,c)[0]
            x['paired'][name]={k:paired_comparison(o[k],baseline[k],ep) for k in ('R@1','hard_accuracy')}
            x['paired_high_both'][name]={k:paired_comparison(o[k][high],baseline[k][high],ep[high]) for k in ('R@1','hard_accuracy')}
            print('control',split,name,x['methods'][name]['R@1'],'high',x['high_both'][name]['R@1'],flush=True)
        for name,source in [('dino_ransac',ra),('sift_ransac',sift)]:
            st=source['status'];x['availability'][name]={'positive_status_counts':{str(k):int((st[ix,pi]==k).sum()) for k in range(4)},'all_pair_status_counts':{str(k):int((st==k).sum()) for k in range(4)}}
        x['availability']['status_definitions']={'dino':['ok','insufficient mutual matches','ransac_failed','reserved'],'sift':['ok','insufficient keypoints','insufficient ratio-test matches','ransac_failed_or_less_than_4_inliers']}
        # Supplemental intersection gallery, explicitly not the primary fixed pool.
        cv=local['candidate_valid']&obs['candidate_valid']&mp['candidate_valid']&(sift['candidate_keypoints']>=4)
        qv=local['query_valid']&obs['query_valid']&mp['query_valid']&(sift['query_keypoints']>=4)&cv[pi]
        qc=[r for r,b in zip(q,qv) if b];cc=[r for r,b in zip(c,cv) if b]
        x['common_reduced_pool']={'queries':len(qc),'candidates':len(cc),'original_queries':len(q),'original_candidates':len(c),'explanation':'only feature-available queries with positive in common gallery; identical reduced gallery for every method, excluded cases retained in primary evaluation', 'methods':{name:summarize(score[qv][:,cv],qc,cc) for name,score in s.items()}}
        # Padding-free supplemental gallery: every retained image has >=95%
        # valid source pixels. The primary gallery above remains unchanged.
        masks=np.load(cache/f'valid_masks_{split}.npz')
        cv95=masks['c'].mean((1,2))>=.95
        qv95=(masks['q'].mean((1,2))>=.95)&cv95[pi]
        q95=[r for r,b in zip(q,qv95) if b];c95=[r for r,b in zip(c,cv95) if b]
        x['valid95_reduced_pool']={'queries':len(q95),'candidates':len(c95),'methods':{name:summarize(score[qv95][:,cv95],q95,c95) for name,score in s.items()}}
        # Query-level outcomes make clustered inference reproducible without GPU.
        saved={'episode':ep,'scene':np.asarray([r['scene_key'] for r in q]),'query_key':np.asarray([r['query_key'] for r in q]),'high_both':high}
        x['scene_cluster_sensitivity']={}
        for name,score in s.items():
            o=query_outcomes(score,q,c)[0]
            for k in ('R@1','R@5','R@10','hard_accuracy','margin','positive_score','negative_score','negative_index'):saved[name+'__'+k]=o[k]
            x['scene_cluster_sensitivity'][name]=clustered_mean_ci(o['R@1'],[r['scene_key'] for r in q])
        np.savez_compressed(OUT/f'query_metric_records_{split}.npz',**saved)
        result['splits'][split]=x;write_json(OUT/'correspondence_controls.json',result)

if __name__=='__main__':main()
