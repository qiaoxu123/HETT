from __future__ import annotations
import math
import numpy as np
import torch
from torch.nn import functional as F

def candidate_scores(logits,candidate_masks):
    """Mean of strongest 25% activation in each candidate polygon."""
    value=logits.float().squeeze();scores=[]
    for mask in candidate_masks:
        m=torch.as_tensor(mask,device=value.device,dtype=torch.bool)
        pixels=value[m]
        if not len(pixels):scores.append(float('-inf'));continue
        scores.append(float(torch.topk(pixels,max(1,len(pixels)//4)).values.mean()))
    return np.asarray(scores,float)

def frame_result(row,logits,label_payload):
    candidates=row['candidates'];masks=[label_payload[f"candidate_{x['landmark_id']}"] for x in candidates];scores=candidate_scores(logits,masks);order=np.argsort(-scores);positive={int(x) for x in row['referenced_landmark_ids']};ranks=[i+1 for i,j in enumerate(order) if int(candidates[j]['landmark_id']) in positive];rank=min(ranks) if ranks else len(candidates)+1
    gt=label_payload['gt_mask'];y,x=np.where(gt>0);pred=int(torch.argmax(logits));h,w=logits.shape[-2:];py,px=divmod(pred,w);pixel_error=math.hypot(px-float(x.mean()),py-float(y.mean()));meters=pixel_error*(2*float(row['altitude_m'])/w)
    pos=max((scores[i] for i,c in enumerate(candidates) if int(c['landmark_id']) in positive),default=float('-inf'));neg=max((scores[i] for i,c in enumerate(candidates) if int(c['landmark_id']) not in positive),default=float('-inf'))
    return {'rank':rank,'top1':float(rank<=1),'top4':float(rank<=4),'top8':float(rank<=8),'mrr':1/rank,'margin':float(pos-neg) if np.isfinite(neg) else float('nan'),'localization_distance_m':meters,'recall@20m':float(meters<=20),'recall@40m':float(meters<=40),'candidate_count':len(candidates)}

def retrieval_result(row,scores):
    """Evaluate GT candidate crops without re-rasterizing overlapping masks."""
    candidates=row['candidates'];scores=np.asarray(scores,float);order=np.argsort(-scores)
    positive={int(x) for x in row['referenced_landmark_ids']}
    ranks=[i+1 for i,j in enumerate(order) if int(candidates[j]['landmark_id']) in positive]
    rank=min(ranks) if ranks else len(candidates)+1
    pos=max((scores[i] for i,c in enumerate(candidates) if int(c['landmark_id']) in positive),default=float('-inf'))
    neg=max((scores[i] for i,c in enumerate(candidates) if int(c['landmark_id']) not in positive),default=float('-inf'))
    predicted=np.asarray(candidates[int(order[0])]['world_xy'],float)
    gt=[np.asarray(c['world_xy'],float) for c in candidates if int(c['landmark_id']) in positive]
    meters=min((float(np.linalg.norm(predicted-x)) for x in gt),default=float('inf'))
    return {'rank':rank,'top1':float(rank<=1),'top4':float(rank<=4),'top8':float(rank<=8),'mrr':1/rank,'margin':float(pos-neg) if np.isfinite(neg) else float('nan'),'localization_distance_m':meters,'recall@20m':float(meters<=20),'recall@40m':float(meters<=40),'candidate_count':len(candidates)}

def aggregate(rows):
    if not rows:return {'samples':0}
    out={'samples':len(rows)}
    for key in ('top1','top4','top8','mrr','margin','recall@20m','recall@40m','localization_distance_m'):
        values=np.asarray([r[key] for r in rows],float)
        out[key if key!='localization_distance_m' else 'mean_localization_distance_m']=float(np.nanmean(values)) if np.isfinite(values).any() else float('nan')
    distances=np.asarray([r['localization_distance_m'] for r in rows],float)
    out['median_localization_distance_m']=float(np.nanmedian(distances)) if np.isfinite(distances).any() else float('nan')
    if any('no_match' in row for row in rows):
        out['no_match_rate']=float(np.mean([float(row.get('no_match',0)) for row in rows]))
    return out

def soft_localization_loss(logits,target):
    logits=F.interpolate(logits,size=(28,28),mode='bilinear',align_corners=False).flatten(1);target=F.interpolate(target.float(),size=(28,28),mode='area').flatten(1);target=target/target.sum(1,keepdim=True).clamp_min(1e-6);return -(target*F.log_softmax(logits,dim=1)).sum(1).mean()

def ranking_loss(logits,candidate_masks,positive_indices,margin=.2):
    losses=[]
    for b in range(len(logits)):
        scores=[]
        for mask in candidate_masks[b]:
            m=F.interpolate(mask[None,None].to(logits.device).float(),size=logits.shape[-2:],mode='nearest')[0,0].bool();v=logits[b,0][m];scores.append(v.topk(max(1,len(v)//4)).values.mean() if len(v) else logits[b,0].min())
        scores=torch.stack(scores);pos=scores[positive_indices[b]].max();negative=torch.ones(len(scores),dtype=torch.bool,device=scores.device);negative[positive_indices[b]]=False
        if negative.any():losses.append(F.relu(scores[negative]-pos+margin).mean())
    return torch.stack(losses).mean() if losses else logits.sum()*0
