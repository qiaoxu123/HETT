import numpy as np

from multiagent.visual_goal.metrics import retrieval_metrics
from multiagent.visual_goal.retrieval import cosine_matrix


def test_cosine_and_metrics_known_case():
    f=np.asarray([[1,0],[0,1],[-1,0]],dtype=np.float32)
    s=cosine_matrix(f,f)
    m=retrieval_metrics(s,np.arange(3),np.ones_like(s,dtype=bool))
    assert m["recall@1"]==1.0
    assert m["same_map_hard_negative_accuracy"]==1.0
    assert m["positive_negative_margin"]>0
