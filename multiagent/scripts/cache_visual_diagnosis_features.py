"""Cache unchanged SigLIP2 global and frozen DINO patch features once per image.
Run from repository root: python -m multiagent.scripts.cache_visual_diagnosis_features.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from multiagent.visual_goal.diagnosis_data import ROOT, DATASET, OUT, load_split, file_hash, write_json
from multiagent.visual_goal.encoder import build_encoder
from multiagent.visual_goal.leakage import assert_visual_only_inputs


def main():
    p=argparse.ArgumentParser(); p.add_argument('--kind',choices=['global','patch'],required=True);p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
    torch.set_num_threads(6)
    cache=OUT/'cache';cache.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'artifacts/visual_goal_abstraction/models/siglip2_partial_trainseen.pt'
    dino='/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056'
    enc=build_encoder('siglip2_partial' if a.kind=='global' else 'dinov2_small','cuda',str(checkpoint) if a.kind=='global' else dino)
    model_hash=file_hash(checkpoint) if a.kind=='global' else file_hash(Path(dino)/'model.safetensors') if (Path(dino)/'model.safetensors').exists() else 'frozen_dinov2_ed25f3a'
    for split in ('val_seen','val_unseen'):
        q,c=load_split(split)
        for role,rows in [('c',c),('q',q)]:
            target=cache/f'{a.kind}_{split}_{role}.npy'
            manifest={'model':model_hash,'metadata_sha256':file_hash(DATASET/f'{"queries" if role=="q" else "templates"}_{split}.jsonl'),'paths':[r['image_path'] for r in rows], 'feature':'global float32' if a.kind=='global' else '256 normalized DINO patches float16; original processor 224 centre crop from square images'}
            meta=target.with_suffix('.json')
            if target.exists() and meta.exists() and json.loads(meta.read_text())==manifest:
                print('reuse',target,flush=True);continue
            result=None
            for i in range(0,len(rows),a.batch_size):
                imgs=[Image.open(DATASET/r['image_path']).convert('RGB') for r in rows[i:i+a.batch_size]]
                inp=enc.processor(images=imgs,return_tensors='pt').to(enc.device);assert_visual_only_inputs(inp)
                with torch.inference_mode():
                    if a.kind=='global':
                        f=enc.model.get_image_features(**inp)
                        if hasattr(f,'pooler_output'):f=f.pooler_output
                    else: f=enc.model(**inp).last_hidden_state[:,1:]
                    f=F.normalize(f.float(),dim=-1).cpu().numpy()
                if result is None:
                    result=np.lib.format.open_memmap(str(target)+'.tmp',mode='w+',dtype=np.float32 if a.kind=='global' else np.float16,shape=(len(rows),)+f.shape[1:])
                result[i:i+len(f)]=f
                if i%320==0: print(a.kind,split,role,i,len(rows),flush=True)
            result.flush();del result
            os.replace(str(target)+'.tmp',target);write_json(meta,manifest)

if __name__=='__main__':main()
