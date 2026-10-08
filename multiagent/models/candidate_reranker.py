"""Residual Top-K reranking over frozen belief, text and landmark features."""
import torch
from torch import nn
from .multi_landmark import candidate_landmark_geometry
from .relative_geometry import dense_relative_geometry

FEATURE_KEYS=('map','language','agent','landmark_geometry','landmark_names','landmark_valid','field')

def extract_candidate_features(belief, ids, language, language_mask, pose, landmarks, relation_head):
    b,k=ids.shape;h=belief.logits.shape[-1]
    spatial=belief.spatial_features.flatten(2).transpose(1,2)
    maps=spatial.gather(1,ids[...,None].expand(-1,-1,spatial.shape[-1]))
    weight=language_mask.to(language.dtype)
    pooled=(language*weight[...,None]).sum(1)/weight.sum(1,keepdim=True).clamp_min(1)
    geom=dense_relative_geometry(pose[:,2:4],pose[:,:2],torch.zeros_like(pose[:,2:4]),torch.zeros(b,device=pose.device,dtype=torch.bool),field_size=h,map_meters=410.)
    geom=geom[:,:7].flatten(2).transpose(1,2).gather(1,ids[...,None].expand(-1,-1,7))
    m=landmarks['landmark_valid'].shape[1]
    masks=landmarks['landmark_text_mask'] & language_mask[:,None,:].bool()
    names=torch.einsum('bml,bld->bmd',masks.to(language.dtype),language)/masks.sum(-1,keepdim=True).clamp_min(1)
    names=torch.where(masks.any(-1)[...,None],names,pooled[:,None,:])
    names=relation_head.name_proj(names)
    lg=candidate_landmark_geometry(landmarks['landmark_xy'],landmarks['landmark_extent'],field_size=h)
    lg=lg.gather(1,ids[...,None,None].expand(-1,-1,m,8))
    # Fixed padding makes feature shards reproducible and independent of batch composition.
    if m>16:raise ValueError('at most 16 landmarks supported')
    valid=torch.zeros(b,16,dtype=torch.bool,device=ids.device);valid[:,:m]=landmarks['landmark_valid']
    ng=lg.new_zeros(b,k,16,8);ng[:,:,:m]=lg
    nnames=names.new_zeros(b,16,96);nnames[:,:m]=names
    raw=belief.logits.flatten(1).gather(1,ids).float()
    normalized=(raw-raw.mean(-1,keepdim=True))/raw.std(-1,unbiased=False,keepdim=True).clamp_min(1e-6)
    rank=torch.linspace(1,0,k,device=ids.device).expand(b,-1)
    return {'map':maps,'language':pooled,'agent':geom,'landmark_geometry':ng,'landmark_names':nnames,'landmark_valid':valid,'field':torch.stack([normalized,rank],-1)}

class CandidateReranker(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.map_proj=nn.Linear(256,hidden);self.language_proj=nn.Linear(768,hidden)
        self.agent_proj=nn.Linear(7,hidden);self.name_proj=nn.Linear(96,hidden)
        self.geometry_proj=nn.Sequential(nn.Linear(8,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.anchor_norm=nn.LayerNorm(hidden)
        self.anchor_score=nn.Linear(hidden,1)
        self.head=nn.Sequential(nn.Linear(hidden*2+2,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Linear(hidden,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,features):
        # Explicit whitelist: labels/GT may never be consumed as features.
        f={k:features[k] for k in FEATURE_KEYS}
        base=self.map_proj(f['map'])+self.language_proj(f['language'])[:,None,:]+self.agent_proj(f['agent'])
        anchors=self.anchor_norm(base[:,:,None,:]+self.name_proj(f['landmark_names'])[:,None,:,:]+self.geometry_proj(f['landmark_geometry']))
        mask=f['landmark_valid'][:,None,:]
        scores=self.anchor_score(anchors).squeeze(-1).masked_fill(~mask,-1e4)
        weights=scores.softmax(-1)*mask.to(scores.dtype)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
        pooled=(anchors*weights[...,None]).sum(2)
        residual=self.head(torch.cat([base,pooled,f['field']],-1)).squeeze(-1)
        return f['field'][...,0]+residual

def install_candidate_features(agent):
    if getattr(agent,'_candidate_feature_hook',None) is None:
        def capture(module,inputs,output):agent._candidate_belief=output
        agent._candidate_feature_hook=agent.vln_model_without_ddp.spatial_belief.register_forward_hook(capture)
