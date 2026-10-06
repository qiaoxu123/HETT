# FPV / Oblique Landmark Appearance Memory

This is an offline visual grounding diagnostic. It does not alter HETT navigation,
train an action policy, or execute rollouts.

## Data audit

The released CityNav package provides Pose5D trajectories, CityRefer geometry,
orthophoto RGB and DSM, but no stored human first-person trajectory frames.
The local Unreal/AirSim renderer was probed at original trajectory poses, but
the CityNav-to-Unreal coordinate registration did not pass the quality gate.
Accordingly, rendered frames are not accepted as valid CityNav observations,
and this branch contains no formal FPV/oblique comparison. The builder can
request front-camera views at pitch 0, -30 and -45 degrees once a calibrated
scene transform is available. Existing teacher-pose orthophoto crops are the
top-down reference, not a paired result from the failed render attempt.

AirSim DepthPerspective is requested, but the initial runtime smoke returned
constant 1.0 for every pixel. Occlusion is therefore not asserted. The
candidate crop masks are pinhole projections of CityRefer footprints and
annotated object heights; they are oracle geometry for crop construction, not
model inputs, and not a replacement for pixel-accurate segmentation.

## Models and memory

The default experiment uses frozen SigLIP2 image/text encoders, RSRefSeg2
official frozen weights, and the previously selected prompter-only checkpoint.
No text encoder, vision encoder, prompter, SAM, or policy is trained in this
experiment. The reference mask is excluded from inference inputs.

History is keyed by landmark ID. Frames are drawn from the same episode at raw
trajectory offsets 0, 1, 3, 5, 7, 9 and 11. Mean, max, fixed quality-weighted
pooling, current-only, past-only and a bounded top-five landmark buffer are
compared. `val_unseen` is held for final evaluation only; thresholds and model
selection are not fitted on it.

## Reproduction

Do **not** treat a dataset produced by the current builder as valid evaluation
data until map-to-renderer registration is calibrated and verified with
landmark overlays plus working depth/visibility checks. The command below is
for reproducing the diagnostic attempt, not a validated benchmark:

```bash
PYTHONPATH=/home/ubuntu5/Workspace/HETT \
  /home/ubuntu5/miniconda3/envs/AirVLN/bin/python \
  scripts/build_fpv_landmark_memory_dataset.py \
  --source-dataset /mnt/windows-data/hett-rsrefseg2/dataset_v1 \
  --data-root /home/ubuntu5/Workspace/HETT/data \
  --output /mnt/windows-data/hett-fpv-landmark-memory/formal_v1 \
  --airsim-port 30001
```

Only after the registration gate passes, inference can use `HF_HOME=/mnt/windows-data/hett-rsrefseg2/hf-cache` and add
`/mnt/windows-data/hett-rsrefseg2/pydeps` to `PYTHONPATH`, then invoke
`scripts/evaluate_fpv_landmark_memory.py` with the resulting manifest and the
source RSRefSeg2 manifest. `--method` is `siglip2`, `rsref_frozen`, or
`rsref_prompter`.

## Current results

Formal evaluation did not run: the renderer alignment gate failed. The small
smoke run is a pipeline check only, not evidence of generalization.
The formal metrics, sample counts, renderer/depth status, and qualitative
limitations are reported in `FPV_LANDMARK_MEMORY_REPORT.md`.
