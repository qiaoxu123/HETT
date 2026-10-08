"""Additional fixed contour-IoU and 4x4 structural-layout controls.
Same image-only transform bank as Chamfer; no held-out parameter search.
"""
import numpy as np
import torch
import cv2
from multiagent.visual_goal.diagnosis_data import OUT
from multiagent.visual_goal.geometry_matching import variants


def main():
    torch.set_num_threads(2)
    for split in ('val_seen','val_unseen'):
        inputs=np.load(OUT/'cache'/f'geometry_inputs_{split}.npz')
        for name in ('observed','map'):
            path=OUT/'cache'/f'geometry_controls_{name}_{split}.npz'
            if path.exists():continue
            q=inputs[name+'_q'];c=inputs[name+'_c'];Q=torch.tensor(q.reshape(len(q),-1),device='cuda');qn=Q.sum(1,keepdim=True)
            qlayout=torch.tensor(np.asarray([cv2.resize(x,(4,4),interpolation=cv2.INTER_AREA).ravel() for x in q]),device='cuda');qlayout=torch.nn.functional.normalize(qlayout,dim=-1)
            iou=np.empty((len(q),len(c)),np.float32);layout=np.empty_like(iou)
            for start in range(0,len(c),16):
                raw=np.asarray([(variants(e)[0]>0).astype(np.float32) for e in c[start:start+16]])
                nc,nv,_=raw.shape;raw=raw.reshape(-1,1024)
                E=torch.tensor(raw,device='cuda');en=E.sum(1)
                elayout=torch.tensor(np.asarray([cv2.resize(x.reshape(32,32),(4,4),interpolation=cv2.INTER_AREA).ravel() for x in raw]),device='cuda');elayout=torch.nn.functional.normalize(elayout,dim=-1)
                for i in range(0,len(q),256):
                    inter=Q[i:i+256]@E.T;score=inter/(qn[i:i+256]+en[None,:]-inter).clamp(min=1)
                    iou[i:i+256,start:start+nc]=score.reshape(-1,nc,nv).max(-1).values.cpu().numpy()
                    ls=qlayout[i:i+256]@elayout.T;layout[i:i+256,start:start+nc]=ls.reshape(-1,nc,nv).max(-1).values.cpu().numpy()
            qv=q.sum((1,2))>=3;cv=c.sum((1,2))>=3
            for s in (iou,layout):s[~qv]=0;s[:,~cv]=0
            np.savez_compressed(path,contour_iou=iou,structural_layout=layout)
            print('geometry controls',split,name,flush=True)

if __name__=='__main__':main()
