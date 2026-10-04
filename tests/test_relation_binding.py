from multiagent.spatial_program.binding import binding_coverage
from multiagent.spatial_program.parser import parse_spatial_program

def test_relation_bound_to_clause_anchor():
    p=parse_spatial_program("Fly past the church, then stop at the building beside the parking lot.")
    near=next(c for c in p.clauses if c.relation=='near')
    names={e.id:e.name for e in p.entities}
    assert names[near.object]=='parking lot'
    assert binding_coverage(p)==1
