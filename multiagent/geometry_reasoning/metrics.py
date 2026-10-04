"""Ranking metrics for fixed B0 candidate pools."""
import numpy as np
def summarize(records):
    out={"samples":len(records)}
    for key in records[0]: out[key]=float(np.mean([row[key] for row in records]))
    return out
