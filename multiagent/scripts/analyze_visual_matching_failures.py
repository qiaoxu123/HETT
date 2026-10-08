"""Overlap-conditioned diagnostics, cluster inference, failure panels and report."""
import json,textwrap
import numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from multiagent.visual_goal.diagnosis_data import ROOT,OUT,DATASET,load_split,write_json
from multiagent.visual_goal.overlap import BINS,bin_mask
from multiagent.visual_goal.failure_analysis import summarize,query_outcomes,paired_comparison,clustered_mean_ci
from multiagent.scripts.evaluate_geometry_vs_rgb import load_scores

MAIN=['global_rgb','local_rgb','sift_ransac','geometry_observed','geometry_map_selfmatch','fusion','oracle_iou']
COLORS=['#355caa','#bd6132','#e0a100','#279668','#9675ab','#cf3a65','#666666']


def balanced_pool(scores,queries,candidates,n=20):
    rng=np.random.default_rng(91);vals=[];eps=[]
    for i,q in enumerate(queries):
        pi=next(j for j,c in enumerate(candidates) if c['scene_key']==q['scene_key'])
        neg=[j for j,c in enumerate(candidates) if c['map_name']==q['map_name'] and j!=pi]
        if len(neg)<n-1:continue
        js=rng.choice(neg,n-1,replace=False);p=scores[i,pi];ns=scores[i,js]
        rank=(ns>p+1e-8).sum();tie=1+(np.abs(ns-p)<=1e-8).sum()
        vals.append(np.clip((1-rank)/tie,0,1));eps.append(q['episode_key'])
    return {'n':len(vals),'pool':n,'R@1':float(np.mean(vals)) if vals else None,'R@1_ci95':clustered_mean_ci(vals,eps)}


def analyze_split(split):
    q,c,s,complete,local,obs,mp=load_scores(split);s['fusion']=np.load(OUT/'cache'/f'fusion_scores_{split}.npy')
    s['sift_ransac']=np.load(OUT/'cache'/f'sift_scores_{split}.npz')['scores']
    ov=np.load(OUT/'cache'/f'overlap_{split}.npz');s['oracle_iou']=ov['iou']
    o={k:query_outcomes(v,q,c)[0] for k,v in s.items()}
    pi=np.asarray([next(j for j,x in enumerate(c) if x['scene_key']==r['scene_key']) for r in q]);ix=np.arange(len(q))
    pos={k:ov[k][ix,pi] for k in ov.files}; ep=np.asarray([r['episode_key'] for r in q]);dist=np.asarray([r['distance_to_goal_m'] for r in q])
    result={'n':len(q),'groups':{},'paired_high_both':{},'hard_negative':{},'controls':{}}
    for key in ('query_coverage','template_coverage','iou'):
        result['groups'][key]={}
        for lo,hi,label in BINS:
            mask=bin_mask(pos[key],lo,hi)
            result['groups'][key][label]={name:summarize(s[name],q,c,mask) for name in MAIN}
    high=(pos['query_coverage']>.5)&(pos['template_coverage']>.5)
    low=(pos['query_coverage']<=.1)|(pos['template_coverage']<=.1)
    zero=pos['intersection_m2']<=1e-8
    for name,mask in [('high_both',high),('low_either',low),('zero_overlap',zero)]:
        result[name]={method:summarize(s[method],q,c,mask,ci=True) for method in MAIN}
    for a,b in [('local_rgb','global_rgb'),('sift_ransac','global_rgb'),('local_nn','global_rgb'),('geometry_observed','global_rgb'),('geometry_map_selfmatch','global_rgb'),('fusion','local_rgb'),('fusion','geometry_observed')]:
        result['paired_high_both'][a+'_minus_'+b]={k:paired_comparison(o[a][k][high],o[b][k][high],ep[high]) for k in ('R@1','hard_accuracy')}
    failed=o['global_rgb']['R@1']<1
    result['failure_accounting']={'full_rgb_failed':int(failed.sum()),'zero_overlap_failures':int((failed&zero).sum()),'zero_overlap_fraction_of_failures':float((failed&zero).sum()/failed.sum()),'low_either_failures':int((failed&low).sum()),'low_either_fraction_of_failures':float((failed&low).sum()/failed.sum()),'high_both_failures':int((failed&high).sum()),'note':'conditional accounting, not an identified causal attribution'}
    cf=np.load(OUT/'cache'/f'global_{split}_c.npy')
    for method in MAIN:
        ni=o[method]['negative_index']; valid=np.isfinite(o[method]['negative_score'])
        row={}
        for label,mask in [('all',valid),('high_query_coverage',valid&(pos['query_coverage']>.7)),('high_template_coverage',valid&(pos['template_coverage']>.7)),('high_both',valid&high)]:
            if not mask.any():row[label]={'n':0};continue
            a=ix[mask];b=ni[mask]
            pp=np.asarray([c[j]['target_xy'] for j in pi[mask]]);np_=np.asarray([c[j]['target_xy'] for j in b])
            row[label]={'n':int(mask.sum()),'hard_acc':float(np.mean(o[method]['hard_accuracy'][mask])), 'negative_query_coverage_mean':float(ov['query_coverage'][a,b].mean()),'negative_template_coverage_mean':float(ov['template_coverage'][a,b].mean()), 'negative_query_coverage_gt50_fraction':float(np.mean(ov['query_coverage'][a,b]>.5)),'negative_template_coverage_gt50_fraction':float(np.mean(ov['template_coverage'][a,b]>.5)),'negative_overlap_ge_positive_iou_fraction':float(np.mean(ov['iou'][a,b]>=pos['iou'][mask]-1e-8)), 'negative_target_center_distance_mean_m':float(np.linalg.norm(pp-np_,axis=1).mean()),'template_global_feature_cosine_mean':float((cf[pi[mask]]*cf[b]).sum(1).mean())}
        result['hard_negative'][method]=row
    masks=np.load(OUT/'cache'/f'valid_masks_{split}.npz');qfrac=masks['q'].mean((1,2));cfrac=masks['c'].mean((1,2))
    fullvalid=qfrac>=.95
    for i,row in enumerate(q):fullvalid[i]&=all(cfrac[j]>=.95 for j,cr in enumerate(c) if cr['map_name']==row['map_name'])
    result['controls']['valid95_all_same_map_candidates']={method:summarize(s[method],q,c,fullvalid) for method in MAIN}
    result['controls']['same_map_pool20']={method:balanced_pool(s[method],q,c) for method in MAIN}
    # Nuisance-only baselines, explicitly diagnostic (not fusion inputs).
    bright=np.asarray([np.asarray(Image.open(DATASET/r['image_path'])).mean()/255 for r in c])
    nuisance={'padding_fraction_only':-np.abs(qfrac[:,None]-cfrac[None,:]),'brightness_only':-np.abs(np.asarray([r['brightness'] for r in q])[:,None]-bright[None,:])}
    result['controls']['nuisance_baselines']={k:summarize(v,q,c) for k,v in nuisance.items()}
    result['controls']['overlap_correlations']={key:float(spearmanr(pos[key],v).statistic) for key,v in [('query_coverage',np.asarray([r['altitude_agl_m'] for r in q])),('template_coverage',np.asarray([r['brightness'] for r in q]))]}
    # Different annotated target scenes can be within one physical template area.
    result['controls']['gallery_counts']={m:sum(x['map_name']==m for x in c) for m in sorted(set(x['map_name'] for x in c))}
    # Figure 1: each coverage definition, fixed bins.
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    for row,key in enumerate(('query_coverage','template_coverage')):
        for col,metric in enumerate(('R@1','hard_accuracy','margin')):
            ax=axes[row,col]
            for method,color in zip(MAIN,COLORS):
                y=[result['groups'][key][label][method].get(metric) for _,_,label in BINS]
                ax.plot(range(6),y,'o-',label=method,color=color)
            ax.set_xticks(range(6),[label for _,_,label in BINS],rotation=25);ax.set_ylabel(metric);ax.set_xlabel(key);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle(split+' / full original gallery');fig.tight_layout();fig.savefig(OUT/f'figure1_overlap_{split}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,('query_coverage','template_coverage','iou')):
        ax.hexbin(dist,pos[key],gridsize=40,mincnt=1,bins='log');ax.set_xlabel('distance to goal (m)');ax.set_ylabel(key);ax.set_title(f'rho={spearmanr(dist,pos[key]).statistic:.3f}')
    fig.tight_layout();fig.savefig(OUT/f'figure2_distance_{split}.png',dpi=150);plt.close(fig)
    # Failure category selection uses original score conditions, not hand-picked success.
    gr=o['global_rgb']['hard_accuracy']>0;ge=o['geometry_observed']['hard_accuracy']>0
    neg=o['global_rgb']['negative_index']
    sim=(cf[pi]*cf[neg]).sum(1)
    categories={'no_overlap_failure':zero&~gr,'high_overlap_global_failure':high&~gr,'rgb_fail_geometry_success':~gr&ge,'geometry_fail_rgb_success':gr&~ge,'both_fail':~gr&~ge,'similar_hard_negative':(~gr)&(sim>.95)}
    result['case_counts']={k:int(v.sum()) for k,v in categories.items()};case_meta=[]
    footprint_records=json.loads((OUT/'cache'/f'footprints_{split}.json').read_text())
    for category,mask in categories.items():
        inds=np.flatnonzero(mask)[:4]
        if not len(inds):continue
        canvas=Image.new('RGB',(1050,410*len(inds)),'white');draw=ImageDraw.Draw(canvas)
        for row,i in enumerate(inds):
            ni=int(neg[i]);yy=row*410
            draw.text((8,yy+4),category+' | '+split,fill='black')
            text='\n'.join(textwrap.wrap(q[i].get('instruction',''),145))[:700];draw.multiline_text((8,yy+22),text,fill='black',spacing=3)
            for col,rec in enumerate((q[i],c[pi[i]],c[ni])):
                im=Image.open(DATASET/rec['image_path']).convert('RGB').resize((224,224));canvas.paste(im,(col*260+8,yy+110))
                draw.text((col*260+8,yy+94),('Query','Positive','Hardest Global Same-map Negative')[col],fill='black')
            desc=f"d={dist[i]:.1f}m | positive QC/TC/IoU={pos['query_coverage'][i]:.3f}/{pos['template_coverage'][i]:.3f}/{pos['iou'][i]:.3f}\nnegative QC/TC/IoU={ov['query_coverage'][i,ni]:.3f}/{ov['template_coverage'][i,ni]:.3f}/{ov['iou'][i,ni]:.3f} | common area={pos['intersection_m2'][i]:.1f} m2\nGlobal pos/neg={s['global_rgb'][i,pi[i]]:.3f}/{s['global_rgb'][i,ni]:.3f} | Local={s['local_rgb'][i,pi[i]]:.3f}/{s['local_rgb'][i,ni]:.3f} | Observed geometry={s['geometry_observed'][i,pi[i]]:.3f}/{s['geometry_observed'][i,ni]:.3f}"
            draw.multiline_text((8,yy+342),desc,fill='black',spacing=3)
            # Map-coordinate footprint inset, visibly marked as offline oracle.
            polys=[footprint_records[j]['valid_polygon'] for j in (i,len(q)+int(pi[i]),len(q)+ni)]
            nonempty=[np.asarray(p) for p in polys if p]
            if nonempty:
                pts=np.concatenate(nonempty);mn=pts.min(0);span=np.maximum(pts.max(0)-mn,1);scale=180/max(span)
                for poly,color in zip(polys,('blue','green','red')):
                    if poly:
                        xy=(np.asarray(poly)-mn)*scale
                        points=[(int(800+x),int(yy+320-y)) for x,y in xy]
                        draw.line(points,fill=color,width=2)
            draw.text((800,yy+90),'Offline footprints',fill='black')
            case_meta.append({'category':category,'query_index':int(i),'query_key':q[i]['query_key'],'positive_index':int(pi[i]),'negative_index':ni})
        canvas.save(OUT/f'figure4_{split}_{category}.jpg',quality=90)
    write_json(OUT/f'failure_cases_{split}.json',case_meta)
    # Save per-query diagnostic measurements, never passed back into matchers.
    with open(OUT/f'query_diagnostics_{split}.jsonl','w') as f:
        for i,row in enumerate(q):
            rec={'query_key':row['query_key'],'episode_key':row['episode_key'],'distance_m':float(dist[i]),'shared_m2':float(pos['intersection_m2'][i]),'query_coverage':float(pos['query_coverage'][i]),'template_coverage':float(pos['template_coverage'][i]),'iou':float(pos['iou'][i]),'global_R1':float(o['global_rgb']['R@1'][i]),'global_hardnegative_index':int(neg[i])}
            f.write(json.dumps(rec)+'\n')
    return result


def main():
    result={}
    for split in ('val_seen','val_unseen'):
        result[split]=analyze_split(split);write_json(OUT/'failure_analysis.json',result)
        print('analysis complete',split,flush=True)
    metrics=json.loads((OUT/'geometry_vs_rgb.json').read_text());oracle=json.loads((OUT/'overlap_oracle.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,split in zip(axes,('val_seen','val_unseen')):
        d=metrics['splits'][split]['methods'];d['sift_ransac']=summarize(np.load(OUT/'cache'/f'sift_scores_{split}.npz')['scores'],*load_split(split));d['oracle_iou']=oracle['splits'][split]['oracles']['iou']
        x=np.arange(len(MAIN));ax.bar(x-.18,[d[k]['R@1'] for k in MAIN],.36,label='global gallery R@1');ax.bar(x+.18,[d[k]['hard_accuracy'] for k in MAIN],.36,label='same-map strict win')
        ax.set_xticks(x,MAIN,rotation=30,ha='right');ax.set_title(split);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'figure3_method_comparison.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
