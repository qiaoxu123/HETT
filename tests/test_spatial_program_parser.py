from multiagent.spatial_program.parser import parse_spatial_program,segment_clauses

def test_clause_roles_binding_and_scope():
    p=parse_spatial_program("Fly past the church and then stop at the second white building on the right.")
    assert len(segment_clauses(p.instruction))==2
    assert next(e for e in p.entities if e.name=='church').role=='reference'
    assert next(e for e in p.entities if e.role=='target').type=='building'
    assert {'past','right_of','ordinal'} <= {c.relation for c in p.clauses}
    ordinal=next(c for c in p.clauses if c.relation=='ordinal')
    assert ordinal.value==2 and ordinal.scope['filters']

def test_parser_never_accepts_hidden_annotation_name():
    p=parse_spatial_program("Stop at the building",["Secret Church"])
    assert all(e.name!='Secret Church' for e in p.entities)
