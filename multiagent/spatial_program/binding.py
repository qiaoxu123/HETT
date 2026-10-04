"""Binding diagnostics and deterministic repair."""
def binding_coverage(program):
    relational=[c for c in program.clauses if c.relation!="ordinal"]
    bound=[c for c in relational if c.object]
    return len(bound)/len(relational) if relational else 1.0

def ambiguous_bindings(program):
    refs=[e for e in program.entities if e.role=="reference"]
    return sum(c.object is None and len(refs)>1 for c in program.clauses)
