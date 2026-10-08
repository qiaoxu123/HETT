"""Reconstruct sampled immutable JPEGs with the recovered crop homographies."""
import json
import cv2
import numpy as np
from PIL import Image
from multiagent.visual_goal.diagnosis_data import ROOT,OUT,DATASET,load_split,write_json


def main():
    results=[]
    for split in ('val_seen','val_unseen'):
        q,c=load_split(split);records=json.loads((OUT/'cache'/f'footprints_{split}.json').read_text());rows=q+c
        bymap={}
        for i,r in enumerate(records):
            key=(r['map_name'],r['role'])
            if key not in bymap:bymap[key]=i
        current=None;rgb=None
        for (m,role),i in sorted(bymap.items()):
            if current!=m:rgb=cv2.imread(str(ROOT/'data/rgbd'/f'{m}.png'));current=m
            H=np.asarray(records[i]['H']);rebuilt=cv2.warpPerspective(rgb,H,(224,224),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
            _,buf=cv2.imencode('.jpg',rebuilt,[cv2.IMWRITE_JPEG_QUALITY,92]);decoded=cv2.imdecode(buf,cv2.IMREAD_COLOR)
            old=cv2.imread(str(DATASET/rows[i]['image_path']))
            error=np.abs(decoded.astype(float)-old.astype(float))
            results.append({'split':split,'map':m,'role':role,'key':records[i]['key'],'mae_after_same_jpeg_encoding':float(error.mean()),'max_error':float(error.max()),'pixel_exact':bool(np.array_equal(decoded,old))})
    write_json(OUT/'coordinate_pixel_audit.json',{'n':len(results),'all_exact':all(x['pixel_exact'] for x in results),'samples':results})
    print('coordinate reconstruction',len(results),'exact',sum(x['pixel_exact'] for x in results),flush=True)

if __name__=='__main__':main()
