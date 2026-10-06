import numpy as np,torch
from multiagent.rsrefseg2_grounding.metrics import candidate_scores,frame_result,retrieval_result
def test_candidate_ranking_and_distance():
 logits=torch.zeros(4,4);logits[:2,:2]=3;m1=np.zeros((4,4),np.uint8);m1[:2,:2]=1;m2=np.zeros_like(m1);m2[2:,2:]=1
 row={'candidates':[{'landmark_id':1},{'landmark_id':2}],'referenced_landmark_ids':[1],'altitude_m':20}
 out=frame_result(row,logits,{'candidate_1':m1,'candidate_2':m2,'gt_mask':m1});assert out['top1']==1 and out['recall@20m']==1 and candidate_scores(logits,[m1,m2])[0]>0

def test_crop_retrieval_keeps_candidate_scores_separate():
 row={'referenced_landmark_ids':[1],'candidates':[{'landmark_id':1,'world_xy':[0,0]},{'landmark_id':2,'world_xy':[30,0]}]}
 out=retrieval_result(row,[2.,1.]);assert out['top1']==1 and out['localization_distance_m']==0
