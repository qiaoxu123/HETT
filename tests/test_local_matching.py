import numpy as np
import torch
from multiagent.visual_goal.local_matching import pair_scores, pool_patches, ransac_correspondences

def test_local_identity_and_permutation_geometry():
    torch.manual_seed(1);x=torch.nn.functional.normalize(torch.randn(1,64,128),dim=-1);valid=torch.ones(1,64,dtype=torch.bool)
    same=pair_scores(x,x,valid,valid); shuffled=pair_scores(x,x[:,torch.randperm(64)],valid,valid)
    assert same['local_rgb'].item()>.99
    assert shuffled['local_rgb'].item()<same['local_rgb'].item()
    assert same['fit_valid'].item()==1

def test_missing_patches_explicit_failure():
    x=torch.zeros(1,64,8);v=torch.zeros(1,64,dtype=torch.bool)
    r=pair_scores(x,x,v,v)
    assert r['fit_valid'].item()==0 and r['local_rgb'].item()==0
    assert ransac_correspondences(x[0].numpy(),x[0].numpy(),v[0].numpy(),v[0].numpy())['status']=='insufficient_correspondences'

def test_spatial_pool_retains_grid():
    assert pool_patches(np.ones((2,256,32))).shape==(2,64,32)

def test_ransac_rotated_patch_correspondence():
    x=np.eye(64,dtype=np.float32)
    perm=np.rot90(np.arange(64).reshape(8,8)).ravel()
    valid=np.ones(64,bool)
    result=ransac_correspondences(x,x[perm],valid,valid)
    assert result['status']=='ok' and result['inlier_ratio']==1


def test_classical_correspondence_and_insufficient_keypoints():
    from multiagent.scripts.evaluate_classic_correspondence import score_pair
    f=np.eye(8,128,dtype=np.float32)
    p=np.float32([[0,0],[20,0],[40,0],[0,20],[20,20],[40,20],[0,40],[40,40]])
    score,status=score_pair(f,p,f,p+10)
    assert score==1 and status==0
    assert score_pair(f[:2],p[:2],f,p)==(0.,1)
