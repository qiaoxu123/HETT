"""Clause-aware deterministic CityNav spatial-program parser."""
from __future__ import annotations
import re
from .entity import extract_entities
from .schema import Axis,Clause,SpatialProgram

RELATIONS=(("left_of",r"\b(?:on|to) the left(?: of)?\b|\bleft of\b"),("right_of",r"\b(?:on|to) the right(?: of)?\b|\bright of\b"),("front_of",r"\bin front of\b|\bahead of\b"),("behind",r"\bbehind\b|\bback of\b"),("near",r"\bnear\b|\bnext to\b|\bbeside\b|\badjacent(?: to)?\b"),("before",r"\bbefore\b"),("after",r"\bafter\b"),("past",r"\bpast\b|\bbeyond\b"),("between",r"\bbetween\b"),("across",r"\bacross\b"),("opposite",r"\bopposite\b"),("along",r"\balong\b"))
ORDINALS=((1,r"\bfirst\b|\b1st\b"),(2,r"\bsecond\b|\b2nd\b"),(3,r"\bthird\b|\b3rd\b"))

def segment_clauses(instruction):
    text=re.sub(r"\s+"," ",instruction.strip())
    parts=re.split(r"\s*(?:,|;|\bthen\b|\band\s+(?=(?:then\s+)?(?:fly|go|turn|stop|land|continue|head|proceed)))\s*",text,flags=re.I)
    return tuple(x.strip(" .") for x in parts if x.strip(" ."))

def _nearest_reference(entities,clause_id,relation):
    refs=[e for e in entities if e.role=="reference"]
    same=[e for e in refs if e.clause_id==clause_id]
    # Sequence relations often live in a motion clause; local reference is strongest.
    choice=(same or refs)
    return choice[-1].id if choice else None

def parse_spatial_program(instruction,referenced_names=()):
    segments=segment_clauses(instruction);entities=extract_entities(segments,referenced_names);clauses=[]
    for ci,text in enumerate(segments,1):
        low=text.casefold();cid=f"c{ci}"
        for relation,pattern in RELATIONS:
            if re.search(pattern,low):
                ref=_nearest_reference(entities,cid,relation)
                clauses.append(Clause(f"r{len(clauses)+1}",text,relation,"target",ref,"travel_axis",confidence=.85,supported=relation not in {"across","opposite"}))
        for value,pattern in ORDINALS:
            if re.search(pattern,low):
                filters=[c.id for c in clauses if c.relation!="ordinal"]
                target=next((e for e in entities if e.role=="target"),None)
                scope={"entity_type":target.type if target else "region","filters":filters,"ordering_origin":"reference"}
                clauses.append(Clause(f"r{len(clauses)+1}",text,"ordinal","target",_nearest_reference(entities,cid,"ordinal"),"travel_axis",value,scope,.8,True))
    refs=[e for e in entities if e.role=="reference"]
    axes=(Axis("travel_axis","start_to_anchor",refs[0].id if refs else None,.9 if refs else .35,"rule"),)
    return SpatialProgram(instruction,segments,entities,tuple(clauses),axes,{"parser":"structured_rule_v1"})
