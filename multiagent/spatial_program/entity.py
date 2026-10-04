"""Entity and role extraction without access to goal annotations."""
from __future__ import annotations
import re
from .schema import Entity

TARGET_TYPES=("building","house","tower","office","hospital","school","college","library","church")
REFERENCE_TYPES=("church","library","college","school","road","street","parking lot","bridge","tower","hospital","field","river","building","house")
ATTRIBUTES=("white","black","gray","grey","red","brown","green","blue","yellow","large","big","small","tall","short","long","wide","narrow","rectangular","square","round","circular","l-shaped","u-shaped")

def _mentions(text, terms):
    return [(m.start(),term) for term in terms for m in re.finditer(rf"\b{re.escape(term)}s?\b",text)]

def extract_entities(segments, referenced_names=()):
    entities=[]; ref_counter=0; target_added=False
    for ci,segment in enumerate(segments):
        low=segment.casefold(); stop=bool(re.search(r"\b(?:stop|land|destination|target|end up|finish)\b",low))
        mentions=_mentions(low,REFERENCE_TYPES)
        for pos,name in sorted(mentions):
            local=low[max(0,pos-35):pos]
            role="target" if stop and (" at " in local or " the " in local) and not target_added else "reference"
            if re.search(r"\b(?:beside|near|after|past|beyond|behind|opposite|across from|right of|left of)\s+(?:the\s+)?$",local): role="reference"
            if role=="target": eid="target"; target_added=True
            else: ref_counter+=1;eid=f"e{ref_counter}"
            attrs=tuple(a for a in ATTRIBUTES if re.search(rf"\b{re.escape(a)}\b",low[max(0,pos-45):pos+len(name)+10]))
            entities.append(Entity(eid,name,name,role,attrs,f"c{ci+1}"))
    # CityRefer annotation links are grounding hints, but only names actually present in text are admitted.
    compact=re.sub(r"[^a-z0-9]","", " ".join(segments).casefold())
    for name in referenced_names:
        if name and re.sub(r"[^a-z0-9]","",name.casefold()) in compact and not any(e.name.casefold()==name.casefold() for e in entities):
            ref_counter+=1;entities.append(Entity(f"e{ref_counter}","landmark",name,"reference",(),"","annotation_link"))
    if not target_added:
        # The final object mention is the target in imperative CityNav instructions.
        candidates=[(i,e) for i,e in enumerate(entities) if e.type in TARGET_TYPES]
        if candidates:
            i,e=candidates[-1];entities[i]=Entity("target",e.type,e.name,"target",e.attributes,e.clause_id,e.source)
    # Stable unique ids after role reassignment.
    out=[];seen=set();n=0
    for e in entities:
        key=(e.name.casefold(),e.role,e.clause_id)
        if key in seen: continue
        seen.add(key)
        if e.role=="reference": n+=1;e=Entity(f"e{n}",e.type,e.name,e.role,e.attributes,e.clause_id,e.source)
        out.append(e)
    if not any(e.role=="target" for e in out):out.insert(0,Entity("target","region","target","target"))
    return tuple(out)
