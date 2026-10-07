"""Answer-blind clause/name alignment for the Gate F diagnostic."""
from __future__ import annotations
import re
from ..anchor_parser import normalise, parse_instruction
from .node_types import NodeKind

STOP={'the','a','an','of','to','with','and','in','on','by','from','near','behind','road','street','building'}

def content_words(text):
    return {w for w in normalise(text).split() if len(w)>2 and w not in STOP}

def bind_phrase(phrase, graph):
    """Return *all* best named map hypotheses; never sees a target/candidate."""
    q=normalise(phrase)
    if not q:return []
    nodes=graph.named()
    exact=[n for n in nodes if n.norm_name==q]
    if exact:return sorted(exact,key=lambda n:n.node_id)
    qwords=content_words(phrase)
    if not qwords:return []
    scored=[]
    for n in nodes:
        words=content_words(n.name)
        score=len(qwords & words)/max(len(qwords),1)
        if score>0:scored.append((score,n))
    if not scored:return []
    best=max(s for s,_ in scored)
    if best<0.5:return []
    return sorted([n for s,n in scored if abs(s-best)<1e-9],key=lambda n:n.node_id)

def explicit_between_second(instruction, first, annotation_phrases):
    """A second anchor only if text explicitly connects A and B after 'between'."""
    m=re.search(r'\bbetween\b(.{0,160})',instruction,re.I)
    if not m or not re.search(r'\band\b',m.group(1),re.I):return None
    before,after=re.split(r'\band\b',m.group(1),maxsplit=1,flags=re.I)
    first_words=content_words(first)
    if first_words and not first_words & content_words(before):return None
    candidates=[]
    for p in annotation_phrases:
        if p==first:continue
        words=content_words(p)
        overlap=len(words & content_words(after))
        if words and overlap>0:candidates.append((overlap/len(words),p))
    if not candidates:return None
    best=max(s for s,_ in candidates)
    bests=[p for s,p in candidates if s==best]
    return bests[0] if len(bests)==1 else None

def annotation_clauses(sample):
    out=[]
    phrases=sample.get('annotation_phrases') or []
    for j,rel in enumerate(sample.get('annotation_relations') or []):
        if not rel or j>=len(phrases):continue
        p=rel.get('phrase')
        if not p:continue
        first=phrases[j]
        second=explicit_between_second(sample['instruction'],first,phrases) if p=='between' else None
        out.append({'phrase':p,'anchor_phrase':first,'second_anchor_phrase':second,'source':'annotation','clause_index':j})
    return out

def text_between_anchors(instruction):
    match=re.search(r'\bbetween\b(.{0,160})',instruction,re.I)
    if not match or not re.search(r'\band\b',match.group(1),re.I):return None
    before,after=re.split(r'\band\b',match.group(1),maxsplit=1,flags=re.I)
    first=re.split(r'[.,;]',before,maxsplit=1)[0].strip(' ,.;')
    second=re.split(r'[.,;]',after,maxsplit=1)[0].strip(' ,.;')
    return (first,second) if first and second else None

def parser_clauses(sample):
    parsed=parse_instruction(sample['instruction'])
    out=[]
    for j,a in enumerate(parsed['anchors']):
        p=a.get('relation');first=a.get('phrase')
        if not p or not first:continue
        pair=text_between_anchors(sample['instruction']) if p=='between' else None
        if pair:first,second=pair
        else:second=None
        out.append({'phrase':p,'anchor_phrase':first,'second_anchor_phrase':second,'source':'parser','clause_index':j})
    return out
