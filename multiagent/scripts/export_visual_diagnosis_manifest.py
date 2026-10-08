"""Save small reproducibility manifests; keep multi-GB caches/checkpoints local."""
import json,platform,subprocess,sys
from pathlib import Path
import cv2,torch,transformers,numpy,rasterio,scipy,PIL,shapely
from multiagent.visual_goal.diagnosis_data import ROOT,OUT,DATASET,write_json,file_hash


def main():
    cache=OUT/'cache';paths=sorted(cache.glob('*'))
    data={'base_commit':'856e8400246730554ee63e39731761e5fdf72973','branch':'2027-CVPR/visual-overlap-geometry-diagnosis','python':sys.version,'platform':platform.platform(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'dependencies':{m.__name__:m.__version__ for m in (cv2,torch,transformers,numpy,rasterio,scipy,PIL,shapely)},'cache':[{'relative_path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in paths if p.is_file()], 'source_metadata':{str(p.relative_to(ROOT)):file_hash(p) for p in DATASET.glob('*.jsonl') if any(s in p.name for s in ('train_seen','val_seen','val_unseen'))},'old_report_sha256':file_hash(ROOT/'VISUAL_GOAL_ABSTRACTION_REPORT.md'),'old_metrics_sha256':file_hash(ROOT/'artifacts/visual_goal_abstraction/eval/full_goal_retrieval.json'),'split_usage':{'train_seen':'original partial fine-tuning only; no new training','val_seen':'fixed model replication, diagnostics and alpha development','val_unseen':'held-out evaluation, all fixed methods reported; no parameter fit','test_unseen':'not accessed'},'cache_policy':'large local caches excluded from git; metrics, query metric records, figures and source scripts committed'}
    # RGB source provenance: hash each orthophoto once, avoid duplicating files.
    maps=json.loads((OUT/'overlap_oracle.json').read_text())['maps']
    data['orthophoto_sources']={m:{ext:file_hash(ROOT/'data/rgbd'/f'{m}.{ext}') for ext in ('png','tif')} for m in maps}
    write_json(OUT/'reproducibility_manifest.json',data)

if __name__=='__main__':main()
