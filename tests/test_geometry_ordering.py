from multiagent.geometry_reasoning.ordering import candidate_order,ordinal_scores
def test_order():
 ranks,_=candidate_order([(3,0),(1,0),(2,0)],(0,0),(1,0));assert ranks==[3,1,2];assert ordinal_scores(ranks,2)[2]>ordinal_scores(ranks,2)[0]
