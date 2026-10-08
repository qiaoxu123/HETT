"""Full-gallery DINO mutual-match RANSAC control, with no pair shortlisting.
Eight CPU workers operate only on cached image patch features. No new encoder.
"""
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import torch
from multiagent.visual_goal.diagnosis_data import OUT,load_split
from multiagent.visual_goal.local_matching import pool_patches,patch_valid,ransac_correspondences

Q=C=QV=CV=None

def block(indices):
    ratios=np.zeros((len(indices),len(C)),np.float32);status=np.zeros_like(ratios,dtype=np.uint8)
    for row,i in enumerate(indices):
        for j in range(len(C)):
            r=ransac_correspondences(Q[i],C[j],QV[i],CV[j]);ratios[row,j]=r['inlier_ratio']
            status[row,j]={'ok':0,'insufficient_correspondences':1,'ransac_failed':2,'degenerate_transform':3}[r['status']]
    return indices,ratios,status


def main():
    global Q,C,QV,CV
    torch.set_num_threads(1)
    for split in ('val_seen','val_unseen'):
        path=OUT/'cache'/f'ransac_scores_{split}.npz'
        if path.exists():continue
        q,c=load_split(split);cache=OUT/'cache';valid=np.load(cache/f'valid_masks_{split}.npz')
        Q=pool_patches(np.load(cache/f'patch_{split}_q.npy',mmap_mode='r')).numpy();C=pool_patches(np.load(cache/f'patch_{split}_c.npy',mmap_mode='r')).numpy()
        QV=patch_valid(valid['q']);CV=patch_valid(valid['c'])
        ratios=np.zeros((len(q),len(c)),np.float32);status=np.zeros_like(ratios,dtype=np.uint8)
        chunks=[np.arange(i,min(i+32,len(q))) for i in range(0,len(q),32)]
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as pool:
            for inds,r,st in pool.map(block,chunks):
                ratios[inds]=r;status[inds]=st
                if inds[0]%640==0:print('RANSAC all candidates',split,int(inds[0]),len(q),flush=True)
        local=np.load(cache/f'local_scores_{split}.npz')
        np.savez_compressed(path,scores=local['mnn']*(.5+.5*ratios),inlier_ratio=ratios,status=status)

if __name__=='__main__':main()
