import numpy as np
from multiagent.rsrefseg2_grounding.dataset import inference_inputs,leakage_audit

def row(split='train_seen'):
 return {'split':split,'episode_id':split+':1','sample_id':split+'-x','map_name':'m','referenced_landmark_ids':[1],'instruction':'church','image':'x.jpg','gt_mask':np.ones((2,2)),'goal':[1,2],'candidate_landmarks':[]}
def test_gt_labels_are_not_in_inference_inputs():
 assert 'gt_mask' not in inference_inputs(row()) and 'goal' not in inference_inputs(row())
def test_forbidden_split_and_overlap_audit():
 report=leakage_audit([row('train_seen'),row('val_unseen')]);assert report['overlap']['train_seen/val_unseen']['samples']==0 and not report['gt_mask_in_inference']
