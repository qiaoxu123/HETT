"""Fixed-frame Grounding DINO probe. Inference sees RGB and text only.

Ground truth geometry is used AFTER prediction, never for box selection.
No SAM, navigation training, full-map image inference, or test_unseen.
"""
import sys,json,time,re,fcntl
from pathlib import Path
import numpy as np
import cv2
import rasterio
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon,box

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from multiagent.cityreferobject import get_city_refer_objects
from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.observation import cropclient
from multiagent.space import Pose4D
from multiagent.defaultpaths import ORTHO_IMAGE_DIR

SOURCE=ROOT.parent/'09-contour-evidence/runs/contour_probe_balanced'
OUT=ROOT/'runs/gdino_fixed30'
OUT.mkdir(parents=True,exist_ok=True)
MODEL='IDEA-Research/grounding-dino-tiny'

def iou(a,b):
    lo=np.maximum(a[:2],b[:2]); hi=np.minimum(a[2:],b[2:])
    inter=np.prod(np.maximum(0,hi-lo))
    return float(inter/max(1e-9,np.prod(np.maximum(0,np.array(a[2:])-a[:2]))+np.prod(np.maximum(0,np.array(b[2:])-b[:2]))-inter))

def noun_prompt(phrase):
    # Only words actually in the instruction-derived target phrase, not GT class.
    vocab=r'building|house|car|vehicle|truck|tree|road|bridge|roof|bus|van|parking lot'
    match=re.search(r'\b('+vocab+r')\b',phrase.lower())
    return (match.group(0) if match else phrase.lower().strip(' .'))+'.'

def main():
    torch.set_num_threads(4)
    frames=json.loads((SOURCE/'results.json').read_text())['frames']
    objects=get_city_refer_objects()
    lookup={}
    for t in load_mturk_trajectories('val_unseen','all',50):
        key=(t.map_name,t.object_id)
        if key not in lookup and objects[t.map_name][t.object_id].processed_descriptions[t.desc_id].landmarks:
            lookup[key]=t.desc_id
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(MODEL,local_files_only=True).eval().to('cuda')
    model.config.disable_custom_kernels=True
    cropclient._raster_cache={}
    rows=[]
    for frame in frames:
        obj=objects[frame['map']][frame['object_id']]
        descid=lookup[(frame['map'],frame['object_id'])]
        phrase=obj.processed_descriptions[descid].target.lower().strip(' .')+'.'
        rgb=Image.open(SOURCE/f"frame_{frame['frame']:02d}.png").convert('RGB')
        # Deliberately complete inference before accessing any target geometry.
        predictions={}
        for mode,prompt in [('category',noun_prompt(phrase)),('target_phrase',phrase)]:
            inputs=processor(images=rgb,text=prompt,return_tensors='pt').to('cuda')
            torch.cuda.synchronize(); start=time.perf_counter()
            with torch.inference_mode(): outputs=model(**inputs)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            result=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
                       box_threshold=.35,text_threshold=.25,target_sizes=[(224,224)])[0]
            order=result['scores'].argsort(descending=True)
            predictions[mode]={'prompt':prompt,'boxes':result['boxes'][order].cpu().tolist(),
                              'scores':result['scores'][order].cpu().tolist(),'seconds':elapsed}
        # Evaluation-only target projection using exactly the dataset crop transform.
        map_name=frame['map']; pose=Pose4D(*frame['pose'])
        if map_name not in cropclient._raster_cache:
            cropclient._raster_cache[map_name]=rasterio.open(ORTHO_IMAGE_DIR/(map_name+'.tif'))
        raster=cropclient._raster_cache[map_name]
        corners=cropclient._compute_view_area_corners_rowcol(map_name,pose)[:,::-1].copy()
        H=cv2.getPerspectiveTransform(corners,np.float32([[0,0],[223,0],[223,223],[0,223]]))
        rc=np.float32([raster.index(x,y)[::-1] for x,y in obj.contour])
        contour=cv2.perspectiveTransform(rc[None],H)[0]
        poly=Polygon(contour)
        if not poly.is_valid: poly=poly.buffer(0)
        clipped=poly.intersection(box(0,0,223,223))
        area=float(clipped.area)
        # Geometric in-view proxy, not manually verified visibility/occlusion.
        status='in_view' if area>=25 and area/max(poly.area,1e-9)>=.25 else ('out_of_view' if area<1 else 'partial_small')
        gt=list(clipped.bounds) if area>0 else None
        point=cv2.perspectiveTransform(np.float32([[raster.index(obj.position.x,obj.position.y)[::-1]]]),H)[0,0]
        row={'frame':frame['frame'],'map':map_name,'object_id':obj.id,'description_id':descid,'phase':frame['phase'],
             'full_description':obj.descriptions[descid],'status':status,'gt_box':gt,'gt_point':point.tolist(),
             'visible_area_pixels':area,'visible_fraction':area/max(poly.area,1e-9),'predictions':predictions}
        for pred in predictions.values():
            pred['top1_iou']=iou(pred['boxes'][0],gt) if pred['boxes'] and gt else 0.
            pred['any_iou']=max([iou(b,gt) for b in pred['boxes']],default=0.) if gt else 0.
        rows.append(row)
        (OUT/'predictions.json').write_text(json.dumps(rows,indent=2))
        print(f"{len(rows)}/30 {status} boxes={[(k,len(v['boxes'])) for k,v in predictions.items()]}",flush=True)
    summary={'model':MODEL,'revision':model.config._commit_hash,'torch':torch.__version__,
             'frames':len(rows),'box_threshold':.35,'text_threshold':.25,
             'in_view':sum(r['status']=='in_view' for r in rows),
             'out_of_view':sum(r['status']=='out_of_view' for r in rows),
             'partial_small':sum(r['status']=='partial_small' for r in rows)}
    for mode in ['category','target_phrase']:
        visible=[r for r in rows if r['status']=='in_view']; absent=[r for r in rows if r['status']=='out_of_view']
        summary[mode]={'top1_iou50_hits':sum(r['predictions'][mode]['top1_iou']>=.5 for r in visible),
            'any_iou50_hits':sum(r['predictions'][mode]['any_iou']>=.5 for r in visible),
            'out_of_view_frames_with_detections':sum(bool(r['predictions'][mode]['boxes']) for r in absent),
            'mean_inference_seconds':float(np.mean([r['predictions'][mode]['seconds'] for r in rows]))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    for name,selected in [('all_frames',rows),('examples',sorted([r for r in rows if r['status']=='in_view'],key=lambda r:r['predictions']['target_phrase']['top1_iou'],reverse=True))]:
        if name=='examples':
            selected=[selected[0],selected[len(selected)//2],selected[-1]]
        fig,axes=plt.subplots(len(selected),2,figsize=(10,4*len(selected)),squeeze=False)
        for axs,r in zip(axes,selected):
            for ax,mode in zip(axs,['category','target_phrase']):
                ax.imshow(Image.open(SOURCE/f"frame_{r['frame']:02d}.png"))
                pred=r['predictions'][mode]
                if r['gt_box']:
                    x,y,x2,y2=r['gt_box'];ax.add_patch(Rectangle((x,y),x2-x,y2-y,fill=False,edgecolor='lime',lw=2))
                for j,b in enumerate(pred['boxes'][:5]):
                    x,y,x2,y2=b;ax.add_patch(Rectangle((x,y),x2-x,y2-y,fill=False,edgecolor='red' if j==0 else 'orange',lw=2 if j==0 else .8))
                import textwrap
                ax.set_title(f"#{r['frame']} {mode}: top1 IoU {pred['top1_iou']:.2f}\n"+'\n'.join(textwrap.wrap(pred['prompt'],48)),fontsize=9)
                ax.set(xlim=(0,224),ylim=(224,0));ax.axis('off')
        fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=110);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    with open(ROOT.parent/'.gpu-validation.lock','a') as lock:
        print('Waiting for shared GPU lock',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
