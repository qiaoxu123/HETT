"""Text-only layout parsing and answer-free local ordering geometry."""
from __future__ import annotations

import math
import re
import numpy as np

ORD = {'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'last':-1}
NUM = r'(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|last)'
DIR = r'(?:left|right|top|bottom)'

def ordinal(word):
    w=word.lower(); return ORD.get(w, int(re.match(r'\d+',w).group()) if re.match(r'\d+',w) else None)

def parse_layout(instruction):
    """No map or answer fields are accepted by this parser."""
    s=instruction.lower(); programs=[]; cues=[]
    for m in re.finditer(rf'\b({NUM})\s+(?:\w+\s+)?from\s+the\s+({DIR})\b',s):
        programs.append({'type':'NTH_FROM_'+m[2].upper(),'index':ordinal(m[1]),'scope':'object_line','span':m.group()})
    for m in re.finditer(rf'\b({NUM})\s+(row|column)\b',s):
        programs.append({'type':'ROW_INDEX' if m[2]=='row' else 'COLUMN_INDEX','index':ordinal(m[1]),'scope':'grid','span':m.group()})
    for m in re.finditer(r'\b(top|bottom)\s+(?:most\s+)?row\b|\b(leftmost|rightmost)\s+column\b',s):
        direction=m[1] or m[2].replace('most','')
        programs.append({'type':('ROW_' if direction in ('top','bottom') else 'COLUMN_')+direction.upper(),'index':1,'scope':'grid','span':m.group()})
    for m in re.finditer(rf'\b({NUM})\s+(?:building|house|object|car)\s+from\s+the\s+(corner|end)\b',s):
        programs.append({'type':'NTH_FROM_CORNER','index':ordinal(m[1]),'scope':'road_sequence','span':m.group()})
    for m in re.finditer(r'\b(?:half\s*way|halfway)\s+(?:down|along)\s+(?:the\s+)?([\w ]{1,40}?\b(?:road|street|avenue))',s):
        programs.append({'type':'HALFWAY_ALONG','index':.5,'scope':'road_sequence','span':m.group(),'reference_phrase':m[1]})
    for m in re.finditer(r'\b(?:at\s+the\s+)?(first|last|end)\s+(?:building|house|car)\s+(?:along|on)\s+(?:the\s+)?([\w ]{1,40}?\b(?:road|street|avenue))',s):
        programs.append({'type':'ROAD_SEQUENCE_ORDER','index':1 if m[1]=='first' else -1,'scope':'road_sequence','span':m.group(),'reference_phrase':m[2]})
    for m in re.finditer(r'\b(?:with|when viewed with)\s+(?:the\s+)?(long|short)\s+side\s+of\s+(.{1,80}?)\s+to\s+the\s+(left|right|top|bottom)\b',s):
        cues.append({'axis_source':'building_long_axis' if m[1]=='long' else 'building_short_axis','reference_phrase':m[2].strip(),'direction':m[3],'span':m.group()})
    for m in re.finditer(r'\bwith\s+([^,.]{1,70}?)\s+at\s+the\s+(top|bottom|left|right)\b',s):
        cues.append({'axis_source':'view_cue','reference_phrase':m[1].strip(),'direction':m[2],'span':m.group()})
    for m in re.finditer(r'\b(?:when\s+viewed\s+(?:aerially\s+)?with|with)\s+([^,.]{1,70}?)\s+to\s+the\s+(left|right|top|bottom)\b',s):
        if not re.search(r'\b(?:long|short)\s+side\b',m[1]):
            cues.append({'axis_source':'view_cue','reference_phrase':m[1].strip(),'direction':m[2],'span':m.group()})
    for m in re.finditer(r'\b(?:first|last|top|bottom)\s+row\s+of\s+[^,.]{1,55}?\s+(?:being|is)\s+at\s+the\s+(top|bottom)\b',s):
        cues.append({'axis_source':'row_axis','reference_phrase':m.group().split(' being')[0],'direction':m[1],'span':m.group()})
    for m in re.finditer(r'\b(?:facing|looking toward|viewed from|from the perspective of)\s+(.{1,60}?)(?:[,.]|$)',s):
        cues.append({'axis_source':'view_cue','reference_phrase':m[1].strip(),'direction':'top','span':m.group()})
    if re.search(r'\b(?:parallel|perpendicular)\s+to\s+(?:the\s+)?\w+\s+road\b|\balong\s+(?:the\s+)?\w+\s+road\b',s):
        cues.append({'axis_source':'road_axis','reference_phrase':'road','direction':'unknown','span':'road axis'})
    return {'target_phrase':instruction.split(',')[0].strip(), 'frame':{'is_explicit':bool(cues),'cues':cues},'layout_programs':programs}

def unit(v):
    a=np.asarray(v,float)[:2]; n=np.linalg.norm(a)
    return a/n if n>1e-9 else None

def footprint_axes(points, minimum_confidence=.12):
    p=np.asarray(points,float)[:,:2]; p=p-p.mean(axis=0)
    if len(p)<3:return None
    _,sing,vt=np.linalg.svd(p,full_matrices=False)
    confidence=(sing[0]**2-sing[1]**2)/(sing[0]**2+sing[1]**2+1e-12)
    if confidence<minimum_confidence:return None
    major=unit(vt[0]);minor=np.array([-major[1],major[0]])
    return {'major':major,'minor':minor,'confidence':float(confidence)}

def oriented_frame(axis,direction):
    """A cue fixes sign from the reference geometry; never inspect the target."""
    a=unit(axis)
    if a is None:return None
    if direction in ('left','bottom'):a=-a
    if direction in ('top','bottom'): y=a;x=np.array([y[1],-y[0]])
    else:x=a;y=np.array([-x[1],x[0]])
    return np.stack([x,y])

def cluster_1d(values,gap=None):
    v=np.asarray(values,float);order=np.argsort(v);n=len(v)
    if n==0:return np.empty(0,int)
    dif=np.diff(v[order]);positive=dif[dif>1e-6]
    if gap is None:gap=float(np.min(positive)*.5) if len(positive) else 1.
    labels=np.empty(n,int);group=0;labels[order[0]]=group
    for k in range(1,n):
        if v[order[k]]-v[order[k-1]]>gap:group+=1
        labels[order[k]]=group
    return labels

def parking_layout(centers,frame=None):
    p=np.asarray(centers,float)[:,:2];z=p-p.mean(axis=0)
    if frame is None:
        _,_,vt=np.linalg.svd(z,full_matrices=False);frame=np.stack([vt[0],vt[1]])
    local=z@np.asarray(frame).T
    pair=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=2)
    np.fill_diagonal(pair,np.inf)
    spacing=float(np.median(np.min(pair,axis=1))) if len(p)>1 else 1.
    gap=max(.75,.4*spacing)
    rows=cluster_1d(local[:,1],gap);cols=cluster_1d(local[:,0],gap)
    return {'frame':np.asarray(frame),'local':local,'rows':rows,'columns':cols}

def soft_rank(predicted,wanted,tau=1.):
    return float(math.exp(-abs(predicted-wanted)/tau))

def score_order(program,local,rows=None,columns=None,tau=1.):
    p=np.asarray(local,float);n=len(p);typ=program['type'];idx=program['index'];scores=np.zeros(n)
    if typ.startswith('NTH_FROM_'):
        axis=0 if typ.endswith(('LEFT','RIGHT')) else 1
        reverse=typ.endswith(('RIGHT','TOP'))
        if rows is not None and axis==0:
            groups=np.asarray(rows)
        elif columns is not None and axis==1:groups=np.asarray(columns)
        else:groups=np.zeros(n,int)
        for group in np.unique(groups):
            ids=np.flatnonzero(groups==group);order=ids[np.argsort(p[ids,axis])]
            if reverse:order=order[::-1]
            for rank,j in enumerate(order,1):scores[j]=soft_rank(rank,idx,tau)
    elif typ.startswith('ROW_') or typ.startswith('COLUMN_'):
        axis=1 if typ.startswith('ROW_') else 0
        reverse=typ.endswith(('TOP','RIGHT')) or (typ=='ROW_INDEX' and program.get('start')=='top')
        labels=np.asarray(rows if axis==1 and rows is not None else columns if axis==0 and columns is not None else cluster_1d(p[:,axis]))
        centers={v:np.mean(p[labels==v,axis]) for v in np.unique(labels)}
        ordered=sorted(centers,key=centers.get,reverse=reverse)
        for j in range(n):scores[j]=soft_rank(ordered.index(labels[j])+1,idx,tau)
    return scores

def road_s(points,polyline):
    p=np.asarray(points,float)[:,:2];line=np.asarray(polyline,float)[:,:2];d=np.diff(line,axis=0);length=np.linalg.norm(d,axis=1)
    cumulative=np.r_[0.,np.cumsum(length)];result=[]
    for q in p:
        options=[]
        for i,vec in enumerate(d):
            if length[i]<1e-9:continue
            t=np.clip((q-line[i])@vec/(length[i]**2),0,1);near=line[i]+t*vec
            options.append((np.linalg.norm(q-near),cumulative[i]+t*length[i]))
        result.append(min(options)[1]/max(cumulative[-1],1e-9) if options else float('nan'))
    return np.array(result)

def road_sequence_score(program,normalized_s,tau=.15):
    s=np.asarray(normalized_s,float);typ=program['type']
    if typ=='HALFWAY_ALONG':return np.exp(-np.abs(s-.5)/tau)
    order=np.argsort(s);rank=np.empty(len(s),int);rank[order]=np.arange(1,len(s)+1)
    wanted=program['index'];rank=len(s)+1-rank if wanted<0 else rank
    return np.exp(-np.abs(rank-abs(wanted))/max(tau,1.))
