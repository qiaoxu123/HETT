"""Optional SIFT + ratio test + RANSAC, full unchanged gallery (no shortlist).
All constants are fixed standard controls, no held-out parameter fitting.
"""
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import multiprocessing as mp
import cv2,numpy as np
from multiagent.visual_goal.diagnosis_data import DATASET,OUT,load_split

F=P=N=CF=CP=CN=None


def extract_one(item):
    path,mask=item;im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    kp,desc=cv2.SIFT_create(nfeatures=128).detectAndCompute(im,mask.astype(np.uint8)*255)
    x=np.zeros((128,128),np.float32);p=np.zeros((128,2),np.float32)
    if desc is None:return x,p,0
    idx=sorted(range(len(kp)),key=lambda j:-kp[j].response)[:128]
    x[:len(idx)]=desc[idx];p[:len(idx)]=[kp[j].pt for j in idx]
    return x,p,len(idx)


def score_pair(f,p,cf,cp):
    if len(f)<4 or len(cf)<4:return 0.,1
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    pairs=matcher.knnMatch(f,cf,k=2)
    good=[a for a,b in pairs if a.distance<.75*b.distance]
    if len(good)<4:return 0.,2
    source=np.float32([p[a.queryIdx] for a in good])/224
    target=np.float32([cp[a.trainIdx] for a in good])/224
    cv2.setRNGSeed(71)
    M,inliers=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=3/224,maxIters=1000,confidence=.99)
    if M is None or inliers is None or inliers.sum()<4:return 0.,3
    return float(inliers.sum()/np.sqrt(len(f)*len(cf))),0


def block(indices):
    scores=np.zeros((len(indices),len(CF)),np.float32);status=np.zeros_like(scores,dtype=np.uint8)
    for row,i in enumerate(indices):
        for j in range(len(CF)):
            scores[row,j],status[row,j]=score_pair(F[i,:N[i]],P[i,:N[i]],CF[j,:CN[j]],CP[j,:CN[j]])
    return indices,scores,status


def main():
    global F,P,N,CF,CP,CN
    cv2.setNumThreads(1);cache=OUT/'cache'
    for split in ('val_seen','val_unseen'):
        output=cache/f'sift_scores_{split}.npz'
        if output.exists():continue
        q,c=load_split(split);valid={k:v for k,v in np.load(cache/f'valid_masks_{split}.npz').items()}
        for role,rows in [('q',q),('c',c)]:
            filename=cache/f'sift_features_{split}_{role}.npz'
            if not filename.exists():
                items=[(DATASET/r['image_path'],m) for r,m in zip(rows,valid[role])]
                with ThreadPoolExecutor(max_workers=8) as pool:features=list(pool.map(extract_one,items))
                np.savez_compressed(filename,f=np.asarray([x[0] for x in features]),p=np.asarray([x[1] for x in features]),n=np.asarray([x[2] for x in features]));del features
        qf=np.load(cache/f'sift_features_{split}_q.npz');cf=np.load(cache/f'sift_features_{split}_c.npz')
        F,P,N=qf['f'],qf['p'],qf['n'];CF,CP,CN=cf['f'],cf['p'],cf['n']
        s=np.zeros((len(q),len(c)),np.float32);st=np.zeros_like(s,dtype=np.uint8)
        chunks=[np.arange(i,min(i+32,len(q))) for i in range(0,len(q),32)]
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as pool:
            for inds,scores,status in pool.map(block,chunks):
                s[inds]=scores;st[inds]=status
                if inds[0]%640==0:print('SIFT full gallery',split,int(inds[0]),len(q),flush=True)
        np.savez_compressed(output,scores=s,status=st,query_keypoints=N,candidate_keypoints=CN)

if __name__=='__main__':main()
