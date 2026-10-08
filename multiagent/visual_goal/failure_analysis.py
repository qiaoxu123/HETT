"""Fixed-gallery score evaluation, fractional tie ranks and episode bootstrap.
AUROC is pooled positives vs *all* same-map negatives, matching the old table.
Hardest-only AUROC and within-query pairwise AUC are separately named.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def query_outcomes(scores, queries, candidates):
    scores=np.asarray(scores,float)
    ids=np.asarray([r['scene_key'] for r in candidates]); maps=np.asarray([r['map_name'] for r in candidates])
    pi=np.asarray([np.flatnonzero(ids==r['scene_key'])[0] for r in queries])
    ps=scores[np.arange(len(queries)),pi]
    same=np.asarray([maps==r['map_name'] for r in queries]);same[np.arange(len(queries)),pi]=False
    neg=np.where(same,scores,-np.inf); ns=neg.max(1); has=same.any(1)
    greater=(scores>ps[:,None]+1e-8).sum(1); ties=(np.abs(scores-ps[:,None])<=1e-8).sum(1)
    o={f'R@{k}':np.clip((k-greater)/np.maximum(ties,1),0,1) for k in (1,5,10)}
    o['hard_accuracy']=np.where(has,ps>ns+1e-8,np.nan).astype(float)
    o['margin']=np.where(has,ps-ns,np.nan)
    o['hard_tie']=np.where(has,np.abs(ps-ns)<=1e-8,np.nan).astype(float)
    o['pairwise_auc']=np.where(has,(((ps[:,None]>scores+1e-8)+.5*(np.abs(ps[:,None]-scores)<=1e-8))*same).sum(1)/np.maximum(same.sum(1),1),np.nan)
    o['positive_score']=ps;o['negative_score']=ns;o['negative_index']=neg.argmax(1)
    o['same_map_random_top1']=1/(same.sum(1)+1)
    return o, same, pi


def clustered_mean_ci(values, episodes, nboot=1000, seed=73):
    values=np.asarray(values,float); good=np.isfinite(values)
    if not good.any():return None
    values=values[good]; ep=np.asarray(episodes)[good]
    _,ix=np.unique(ep,return_inverse=True);n=ix.max()+1
    sums=np.bincount(ix,weights=values,minlength=n); counts=np.bincount(ix,minlength=n)
    rng=np.random.default_rng(seed);bs=[]
    for _ in range(nboot):
        sample=rng.integers(n,size=n);bs.append(sums[sample].sum()/counts[sample].sum())
    return [float(x) for x in np.quantile(bs,[.025,.975])]


def pooled_auc_ci(scores, same, pi, episodes, nboot=300, seed=73):
    # Cluster-bootstrap a fixed 1024-bin empirical score CDF. Point AUROC is
    # exact; quantized AUROC CI avoids sorting ~1M pairs 300 times per method.
    s=np.asarray(scores,float);ps=s[np.arange(len(s)),pi]
    lo=min(float(ps.min()),float(s[same].min()));hi=max(float(ps.max()),float(s[same].max()))
    ix=np.clip(((s-lo)/max(hi-lo,1e-12)*1023).astype(int),0,1023)
    _,ei=np.unique(episodes,return_inverse=True);ne=ei.max()+1
    ph=np.zeros((ne,1024));nh=np.zeros_like(ph)
    np.add.at(ph,(ei,ix[np.arange(len(s)),pi]),1)
    row,col=np.where(same);np.add.at(nh,(ei[row],ix[row,col]),1)
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(nboot):
        weights=np.bincount(rng.integers(ne,size=ne),minlength=ne)
        p=weights@ph;n=weights@nh
        vals.append(float(np.sum(p*(np.cumsum(n)-.5*n))/max(p.sum()*n.sum(),1)))
    return np.quantile(vals,[.025,.975]).tolist()


def summarize(scores, queries, candidates, selected=None, ci=False):
    if selected is not None:
        scores=np.asarray(scores)[selected];queries=[q for q,b in zip(queries,selected) if b]
    if not queries:return {'n':0}
    o,same,pi=query_outcomes(scores,queries,candidates)
    ep=[r['episode_key'] for r in queries]
    result={'n':len(queries),'episodes':len(set(ep)), 'candidates':len(candidates)}
    for key in ('R@1','R@5','R@10','hard_accuracy','margin','hard_tie','pairwise_auc','same_map_random_top1'):
        valid=np.isfinite(o[key]);result[key]=float(o[key][valid].mean()) if valid.any() else None
        if ci:result[key+'_ci95']=clustered_mean_ci(o[key],ep)
    ps=o['positive_score'];ns=o['negative_score'];valid=np.isfinite(ns)
    result['auroc']=float(roc_auc_score(np.r_[np.ones(len(ps)),np.zeros(same.sum())],np.r_[ps,np.asarray(scores)[same]])) if same.any() else None
    result['hardest_only_auroc']=float(roc_auc_score(np.r_[np.ones(valid.sum()),np.zeros(valid.sum())],np.r_[ps[valid],ns[valid]])) if valid.any() else None
    result['mean_distance_m']=float(np.mean([r['distance_to_goal_m'] for r in queries]))
    if ci and same.any():result['auroc_ci95']=pooled_auc_ci(scores,same,pi,ep)
    return result


def paired_comparison(a,b,episodes,nboot=1000):
    delta=np.asarray(a)-np.asarray(b);good=np.isfinite(delta);delta=delta[good];episodes=np.asarray(episodes)[good]
    if len(delta)==0:return {'n':0}
    _,ix=np.unique(episodes,return_inverse=True);sums=np.bincount(ix,weights=delta)
    rng=np.random.default_rng(19);obs=abs(sums.sum());exceed=0
    for _ in range(4999):exceed+=abs((rng.choice([-1,1],size=len(sums))*sums).sum())>=obs-1e-12
    return {'n':len(delta),'episodes':len(sums),'mean_delta':float(delta.mean()),'ci95':clustered_mean_ci(delta,episodes,nboot), 'two_sided_cluster_signflip_p':(exceed+1)/5000}
