#!/usr/bin/env python3
"""Collate immutable RSRefSeg2 diagnostic runs into committed JSON/CSV."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

RUNS={
 'Current SigLIP':('baseline_siglip_s0_r2','dense'),
 'Whole RGB SigLIP2':('baseline_siglip2_s0_r4','dense'),
 'Region-aware SigLIP2':('baseline_siglip2_s0_r4','region'),
 'RSRefSeg2 Frozen':('fixedpad_experiment_a_frozen_s0','full'),
 'RSRefSeg2 Prompter':('fixedpad_experiment_b_prompter_eval_s0','full'),
 'RSRefSeg2 Vision-LoRA':('fixedpad_experiment_c_vision_lora_eval_s0','full'),
 'GT reference mask (oracle)':('oracle_gt_reference_mask','gt_reference_mask'),
}
CROPS={
 'Current SigLIP crop':('baseline_siglip_s0_r2','region'),
 'SigLIP2 crop':('baseline_siglip2_s0_r4','region'),
 'RSRefSeg2 Frozen crop':('fixedpad_gtcrop_rsref_frozen_s0','full'),
 'RSRefSeg2 Prompter crop':('fixedpad_gtcrop_rsref_prompter_s0','full'),
 'RSRefSeg2 Vision-LoRA crop':('fixedpad_gtcrop_rsref_vision_lora_s0','full'),
}
KEYS=('top1','top4','top8','mrr','margin','recall@20m','recall@40m','mean_localization_distance_m','median_localization_distance_m')

def metric(root,name,variant,split):
 data=json.loads((root/name/'artifacts/metrics.json').read_text());return data['splits'][split][variant],data

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--dataset-stats',type=Path,required=True);p.add_argument('--output-json',type=Path,required=True);p.add_argument('--output-csv',type=Path,required=True);a=p.parse_args()
 result={'dataset':json.loads(a.dataset_stats.read_text()),'full_image':{},'gt_crop':{},'training':{},'language_ablation':{},'static_b0_context':{'val_unseen':{'r@1/20':.2143,'r@4/20':.4698,'r@8/20':.6596,'r@16/20':.8057,'top1_distance_m':49.20},'perfect_referenced_landmark_geometry_not_fair_grounding_baseline':{'r@1/20':.6974,'r@4/20':.8024}},'experiment_d':{'status':'skipped','reason':'Vision-LoRA improved prompter-only by only 0.56 val_unseen Top-1 point and did not robustly clear the gate across both splits/metrics; SAM cannot answer the remaining GT-crop identity failure.'},'test_unseen_read':False}
 rows=[]
 for method,(run,variant) in RUNS.items():
  result['full_image'][method]={}
  for split in ('val_seen','val_unseen'):
   values,meta=metric(a.runs,run,variant,split);result['full_image'][method][split]=values
   rows.append({'diagnostic':'full_image','method':method,'split':split,**{k:values.get(k) for k in KEYS},'seconds':meta.get('seconds'),'peak_gpu_memory_bytes':meta.get('peak_gpu_memory_bytes')})
 for method,(run,variant) in CROPS.items():
  result['gt_crop'][method]={}
  for split in ('val_seen','val_unseen'):
   values,meta=metric(a.runs,run,variant,split);result['gt_crop'][method][split]=values
   rows.append({'diagnostic':'gt_crop','method':method,'split':split,**{k:values.get(k) for k in KEYS},'seconds':meta.get('seconds'),'peak_gpu_memory_bytes':meta.get('peak_gpu_memory_bytes')})
 for short,run in [('prompter','fixedpad_experiment_b_prompter_s0'),('vision_lora','fixedpad_experiment_c_vision_lora_s0')]:
  result['training'][short]=json.loads((a.runs/run/'artifacts/metrics.json').read_text())
 for short,run in [('prompter','fixedpad_experiment_b_prompter_eval_s0'),('vision_lora','fixedpad_experiment_c_vision_lora_eval_s0')]:
  data=json.loads((a.runs/run/'artifacts/metrics.json').read_text());result['language_ablation'][short]={s:{k:v for k,v in data['splits'][s].items() if k!='groups'} for s in ('val_seen','val_unseen')}
 result['decision']={'classification':'mixed problem','localization_signal':'CityNav adaptation improves full-image candidate Top-1, so localization is a real bottleneck.','observability_signal':'Even GT crops peak at 47.4% Top-1 on val_unseen, so appearance is not sufficiently discriminative.','language_signal':'name/category outperforms full instruction on val_unseen, so relation/context language does not transfer reliably.','integrate_into_hett':False}
 a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(result,indent=2)+'\n')
 with a.output_csv.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');writer.writeheader();writer.writerows(rows)
if __name__=='__main__':main()
