"""Parser comparison metrics."""
from __future__ import annotations
import numpy as np

def _f1(pred,gold):
    p=set(pred);g=set(gold);tp=len(p&g);precision=tp/len(p) if p else float(not g);recall=tp/len(g) if g else float(not p)
    return 2*precision*recall/(precision+recall) if precision+recall else 0.
def parser_metrics(pairs):
    rows=[]
    for pred,gold in pairs:
        pe={(e.type,e.role) for e in pred.entities};ge={(e.type,e.role) for e in gold.entities};pr={(c.relation) for c in pred.clauses};gr={(c.relation) for c in gold.clauses}
        pb={(c.relation,c.object) for c in pred.clauses};gb={(c.relation,c.object) for c in gold.clauses}
        rows.append({"entity_f1":_f1(pe,ge),"role_accuracy":float(pe==ge),"relation_f1":_f1(pr,gr),"binding_accuracy":_f1(pb,gb),"axis_accuracy":float([a.definition for a in pred.axes]==[a.definition for a in gold.axes]),"ordinal_scope_accuracy":float([c.scope for c in pred.clauses if c.relation=='ordinal']==[c.scope for c in gold.clauses if c.relation=='ordinal']),"exact_match":float(pred.canonical_json()==gold.canonical_json())})
    return {"samples":len(rows),**({k:float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {})}
