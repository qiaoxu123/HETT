from __future__ import annotations
import json,re
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import Dataset

def language(row,mode):
    if mode=='full':return row['instruction']
    if mode=='landmark_only':return row.get('referenced_phrase') or ', '.join(row.get('referenced_landmark_names',()))
    if mode=='name_category':
        names=', '.join(row.get('referenced_landmark_names',()));cats=', '.join(sorted({x['category'] for x in row.get('positives',())}));return f"{names} {cats}".strip()
    if mode=='no_spatial':return re.sub(r"\b(left|right|behind|front|near|beside|next to|past|after|before|between|across|along|opposite)\b","",row['instruction'],flags=re.I)
    raise ValueError(mode)

class GroundingDataset(Dataset):
    def __init__(self,root,split,language_mode='full',limit=0):
        self.root=Path(root);self.rows=[json.loads(x) for x in (self.root/'manifest.jsonl').read_text().splitlines() if json.loads(x)['split']==split];self.rows=self.rows[:limit or None];self.language_mode=language_mode
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];bgr=cv2.imread(row['image']);image=torch.from_numpy(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1).float()/255;payload=np.load(row['label_file']);gt=torch.from_numpy(payload['gt_mask']).float();candidate=[];positive=[]
        positives={int(x) for x in row['referenced_landmark_ids']}
        for j,c in enumerate(row['candidates']):candidate.append(torch.from_numpy(payload[f"candidate_{c['landmark_id']}"]).float());positive.extend([j] if int(c['landmark_id']) in positives else [])
        return {'image':image,'text':language(row,self.language_mode),'target':gt,'candidate_masks':candidate,'positive_indices':positive,'row':row,'payload':{k:payload[k] for k in payload.files}}
def collate(batch):return {'images':torch.stack([x['image'] for x in batch]),'texts':[x['text'] for x in batch],'targets':torch.stack([x['target'] for x in batch])[:,None],'candidate_masks':[x['candidate_masks'] for x in batch],'positive_indices':[x['positive_indices'] for x in batch],'rows':[x['row'] for x in batch],'payloads':[x['payload'] for x in batch]}
