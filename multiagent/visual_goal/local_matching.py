"""Local DINO patch correspondences; inputs are features and image-valid masks only.
8x8 spatial cells pool 2x2 native 16x16 tokens, never whole-image pooling.
Mutual matches fit a least-squares 2D similarity transform (not RANSAC).
A separate RANSAC diagnostic is available for >=4 correspondences.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def pool_patches(features):
    x=torch.as_tensor(np.asarray(features).astype(np.float32))
    n,p,d=x.shape;s=int(p**.5)
    if s*s!=p or s%2:raise ValueError('expected even square patch grid')
    x=x.reshape(n,s,s,d).reshape(n,s//2,2,s//2,2,d).mean((2,4))
    return F.normalize(x.reshape(n,-1,d),dim=-1)


def patch_valid(masks):
    # Exact default DINO processor: 224 image ->256 resize ->224 centre crop.
    out=[]
    for m in masks:
        resized=cv2.resize(m.astype(np.float32),(256,256),interpolation=cv2.INTER_LINEAR)[16:240,16:240]
        out.append(cv2.resize(resized,(8,8),interpolation=cv2.INTER_AREA).reshape(-1)>.95)
    return np.asarray(out)


def pair_scores(q,c,qvalid,cvalid):
    """Torch aligned pair batches [B,64,D], supports broadcasted pair dimensions."""
    sim=q@c.transpose(-1,-2)
    allowed=qvalid[..., :,None]&cvalid[...,None,:]
    sim=sim.masked_fill(~allowed,-2)
    v,match=sim.max(-1); reverse=sim.argmax(-2)
    indices=torch.arange(q.shape[-2],device=q.device).expand_as(match)
    mutual=(reverse.gather(-1,match)==indices)&qvalid&(v>.6)
    nn=((v.clamp(min=0)*qvalid).sum(-1)/qvalid.sum(-1).clamp(min=1)+ (sim.max(-2).values.clamp(min=0)*cvalid).sum(-1)/cvalid.sum(-1).clamp(min=1))/2
    side=int(q.shape[-2]**.5)
    y,x=torch.meshgrid(torch.linspace(0,1,side,device=q.device),torch.linspace(0,1,side,device=q.device),indexing='ij')
    xy=torch.stack([x,y],-1).reshape(-1,2)
    source=xy.expand(match.shape+(2,));target=xy[match]
    w=mutual.float();n=w.sum(-1);den=n.clamp(min=1)
    mx=(source*w[...,None]).sum(-2)/den[...,None]; my=(target*w[...,None]).sum(-2)/den[...,None]
    X=source-mx[...,None,:];Y=target-my[...,None,:]
    xx=(w*X.square().sum(-1)).sum(-1).clamp(min=1e-8)
    a=(w*(X*Y).sum(-1)).sum(-1)/xx
    b=(w*(X[...,0]*Y[...,1]-X[...,1]*Y[...,0])).sum(-1)/xx
    pred=torch.stack([a[...,None]*X[...,0]-b[...,None]*X[...,1],b[...,None]*X[...,0]+a[...,None]*X[...,1]],-1)
    error=(pred-Y).square().sum(-1).sqrt()
    inliers=((error<.15)&mutual).sum(-1)/den
    scale=(a*a+b*b).sqrt();ok=(n>=4)&(scale>=.1)&(scale<=10)
    consistency=torch.where(ok,inliers,torch.zeros_like(inliers))
    mnn=(v.clamp(min=0)*w).sum(-1)/den
    # Fraction of visible cells discourages one accidental high-similarity match.
    support=(n/torch.minimum(qvalid.sum(-1),cvalid.sum(-1)).clamp(min=1)).sqrt()
    score=mnn*support*(.5+.5*consistency)
    return {'nn':nn,'mnn':mnn*support,'local_rgb':score,'matches':n,'fit_valid':ok.float(),'consistency':consistency}


def ransac_correspondences(q,c,qvalid,cvalid):
    s=np.asarray(q)@np.asarray(c).T;s[~(qvalid[:,None]&cvalid[None,:])]=-2
    match=s.argmax(1);reverse=s.argmax(0);keep=(reverse[match]==np.arange(len(q)))&(s.max(1)>.6)&qvalid
    n=int(keep.sum())
    if n<4:return {'matches':n,'status':'insufficient_correspondences','inlier_ratio':0.}
    side=int(len(q)**.5);y,x=np.mgrid[:side,:side];xy=np.stack([x,y],-1).reshape(-1,2).astype(np.float32)/(side-1)
    cv2.setRNGSeed(71)
    M,inliers=cv2.estimateAffinePartial2D(xy[keep],xy[match[keep]],method=cv2.RANSAC,ransacReprojThreshold=.1,maxIters=1000,confidence=.99)
    if M is None:return {'matches':n,'status':'ransac_failed','inlier_ratio':0.}
    return {'matches':n,'status':'ok','inlier_ratio':float(inliers.mean())}
