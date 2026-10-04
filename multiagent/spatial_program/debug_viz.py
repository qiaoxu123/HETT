"""Debug-view payload helpers kept separate from inference."""
def candidate_table(points, probabilities, clause_scores=None):
    clause_scores=clause_scores or {}
    return [{"rank":i+1,"world_xy":list(map(float,p)),"b0":float(b),
             "constraints":{k:float(v[i]) for k,v in clause_scores.items()}}
            for i,(p,b) in enumerate(zip(points,probabilities))]
