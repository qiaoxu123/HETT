import json
from multiagent.spatial_program.schema import Axis,Clause,Entity,SpatialProgram

def test_schema_roundtrip_is_deterministic():
    p=SpatialProgram("go past church",("go past church",),(Entity("e1","church","church","reference"),Entity("target","building","building","target")),(Clause("c1","past church","past",object="e1",axis="travel"),),(Axis("travel","start_to_anchor","e1",.9),))
    assert SpatialProgram.from_dict(json.loads(p.canonical_json())).canonical_json()==p.canonical_json()
