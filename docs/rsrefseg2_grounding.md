# RSRefSeg2 grounding diagnostic

This branch is an offline referenced-landmark grounding diagnostic. It does not
modify HETT's controller, policy, rollout, or navigation losses.

## Provenance

- Official source: `KyanChen/RSRefSeg2`, commit
  `b717f8dbbd9dbb67cb711e71cff49e03ba53c258`.
- Official RefSegRS checkpoint SHA-256:
  `6018af204120cf14e7555ff067f6fa4a921eba084c33be308bc65bc540f826ee`.
- The official `CascadedPrompter` classes are loaded directly from the pinned
  checkout by `official_adapter.py`; the implementation is not a look-alike
  network. OpenMMLab decorators and its base class are removed by an AST loader
  because they are irrelevant before SAM.
- The backbone is the official SigLIP2 SO400M patch16-512 configuration. The
  coarse output is `dense_prompts`, before SAM.
- Dependencies are isolated with `--pythonpath`; HETT's Python environment is
  not overwritten.

## Data boundary

Images are the existing yaw-dependent, teacher-trajectory CityNav top-down
RGB crops. They are real observations at the recorded pose, not
candidate-centred renders. They are **not genuine oblique/forward FPV**.

Projected candidate masks come from global landmark geometry and are used only
to pool an inference-time heatmap into candidate scores. The union of the true
referenced masks, referenced IDs, and goal are labels only. `inference_inputs`
has an allow-list and tests guard this boundary.

Only `train_seen`, `val_seen`, and `val_unseen` are accepted. `test_unseen` is
rejected by dataset and CLI guards. Episodes/images never cross splits.
`train_seen` and `val_seen` intentionally share 13 named anchors because the
official seen split shares maps; `val_unseen` shares none.

## Commands

Dataset construction:

```bash
/home/ubuntu5/miniconda3/envs/AirVLN/bin/python \
  scripts/build_rsrefseg2_citynav_dataset.py \
  --source-dataset /mnt/windows-data/hett-scene-evidence/phase2_teacher_rgb_s0_20261004/artifacts \
  --output /mnt/windows-data/hett-rsrefseg2/dataset_v1
```

Formal runs use `scripts/supervise_experiment.py` with phase
`rsrefseg2-train` or `rsrefseg2-eval`, the isolated dependency directory
`/mnt/windows-data/hett-rsrefseg2/pydeps`, and the official source/checkpoint
paths above. See each run's `commands.json` for the exact immutable command.

## Loss and adaptation

The coarse 28x28 target is trained with:

```text
L = localization + 0.5 * hard-negative ranking + 0.2 * InfoNCE
```

82.88% of frames contain at least one same-scene hard negative, and all
candidate-ranking negatives come from the same scene. The text encoder is
always frozen. Prompter-only trains the official cascaded prompter and
localization projections. Vision-LoRA additionally trains the official vision
LoRA adapters; it never full-finetunes the vision backbone. SAM is not used.

The benchmark is conditional on the referenced landmark being visible. The
dataset statistics report the excluded invisible fraction, which must be
considered when interpreting deployment value.
