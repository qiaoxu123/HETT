# Spatial Program Reasoner

This branch evaluates whether explicit geometry stated in a CityNav instruction can rerank the frozen Static B0 candidate pool. It never reads RGB, predicts actions, or runs a navigation rollout.

## Inference boundary

Allowed inputs are the instruction, instruction-linked referenced landmark records, map contours/objects, start pose/yaw, and B0 Top-16 candidates. Target coordinates are used only after inference to compute metrics and draw debug figures. `test_unseen` is rejected by the experiment entry point.

The pipeline is clause segmentation → entity roles → relation binding → reference-axis resolution → ordinal scope → sequential soft execution. Every relation is attached to a subject and reference entity. The preferred axis is start-to-anchor; start heading, landmark contour PCA, local road PCA, and anchor-pair axes are retained as auditable alternatives.

## Gold-set status

`analysis/spatial_program_gold.json` is deliberately a review queue, not automatically asserted as human gold. Records contain `human_reviewed=false`. `evaluate_gold_spatial_program.py` refuses to report P7 until at least 150 records have been independently reviewed. `evaluate_spatial_program_parser.py --allow-silver` is only a parser consistency diagnostic and labels its output `silver_self_agreement`.

## Reproduction

```bash
python scripts/build_spatial_program_gold.py --count 200
python scripts/analyze_spatial_programs.py --output runs/spatial_program_s0/phase1_statistics.json
python scripts/supervise_experiment.py \
  --run-dir runs/spatial_program_eval_s0_20261004_r3 \
  --python /home/ubuntu5/miniconda3/envs/AirVLN/bin/python \
  --phase spatial-program-reasoner \
  --b0-cache-root /mnt/windows-data/hett-dynamic-belief/cache
python scripts/visualize_spatial_program.py \
  --debug-records runs/spatial_program_eval_s0_20261004_r3/artifacts/debug_records.json \
  --output runs/spatial_program_eval_s0_20261004_r3/artifacts/debug_viz
```

All model-selection weights are selected on `val_seen`; `val_unseen` is frozen evaluation only.
