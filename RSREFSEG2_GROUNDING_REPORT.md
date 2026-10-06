# RSRefSeg2-style CityNav grounding diagnostic

## Executive conclusion

The result is a **mixed problem**, dominated by visual observability and
appearance ambiguity, with a secondary localization/domain-adaptation problem
and weak generalization of complex instruction language.

Official RSRefSeg2 transfers worse than the SigLIP2 full-image baseline on
`val_unseen` (32.21% versus 38.95% Top-1). CityNav prompter adaptation improves
this to 48.60% (+9.65 points), while Vision-LoRA reaches 49.16% (+10.21).
LoRA adds only 0.56 point over Prompter-only. This is useful localization signal, but it misses the
pre-registered “clearly effective” requirement on `val_unseen` by a small
margin and is not robust enough to integrate into HETT.

More importantly, giving the model the **GT candidate crops** does not solve
the task: the best `val_unseen` Top-1 is only 47.36%. Fine-tuning hurts both
seen and unseen crop retrieval relative to the frozen model. Therefore the remaining failure is
not simply coarse localization; the top-down RGB appearance often cannot
distinguish same-scene named landmarks.

Experiment D (SAM) was skipped by the planned gate. Vision-LoRA beat
Prompter-only by only 0.56 unseen Top-1 point, not a robust gain, and a finer
boundary cannot repair poor GT-crop identity retrieval. No controller,
navigation rollout, or `test_unseen` was run.

## 1. Dataset and leakage audit

| Split | Visible samples | Referenced landmark visible | Rejected as invisible |
|---|---:|---:|---:|
| train_seen | 1,940 | 85.99% | 316 |
| val_seen | 832 | 74.82% | 280 |
| val_unseen | 891 | 80.20% | 220 |

- Total diagnostic samples: 3,663.
- Mean visible candidates/frame: 7.72.
- 82.88% of frames contain at least one same-scene hard negative; every
  candidate-ranking negative is drawn from the same scene.
- Episode/sample overlap across splits: zero.
- Referenced-object overlap with `val_unseen`: zero.
- `train_seen`/`val_seen` share 13 anchors by official seen-split design; no
  images or episodes overlap.
- GT mask and GT goal are absent from the model input allow-list.
- These metrics are visibility-conditioned. On all `val_unseen` frames, a
  rough Prompter Top-1 ceiling from visibility alone is
  `0.8020 * 0.4871 = 39.1%` before considering invisible-frame handling.

The observations are yaw-dependent top-down orthophoto crops at real teacher
trajectory poses. They are not candidate-centred crops and are not genuine
oblique/FPV images. That camera limitation is part of the observability result.

## 2. Official implementation fidelity

The experiment uses the official RSRefSeg2 source at commit
`b717f8dbbd9dbb67cb711e71cff49e03ba53c258` and official RefSegRS weights
(SHA-256 `6018af...26ee`). The official SigLIP2 SO400M encoders and
`CascadedPrompter` are loaded directly. The evaluated heatmap is the official
`dense_prompts` output before SAM.

The complete model has 1,227,539,954 parameters. Prompter-only trains
82,773,312; Vision-LoRA trains 86,966,080. The text encoder is frozen in every
run. Training uses the requested localization, margin-ranking, and InfoNCE
losses with weights 1.0/0.5/0.2.

## 3. Full-image grounding

Candidate Top-K scores pool the predicted coarse heatmap over all global
landmark candidate polygons. Candidate geometry is available at inference; the
model is never told which polygon is the GT reference.

### val_seen

| Method | Trainable | Top-1 | Top-4 | Top-8 | MRR | Margin | R@20m | R@40m | Median dist |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current SigLIP | none | 32.21 | 67.55 | 84.50 | .504 | -.018 | 11.78 | 29.69 | 70.94m |
| Whole RGB SigLIP2 | none | 40.02 | 87.62 | 96.27 | .606 | -.006 | 15.38 | 36.90 | 53.00m |
| Region-aware SigLIP2 | none | 44.71 | 69.11 | 85.82 | .579 | -.008 | 14.90 | 30.17 | 69.74m |
| RSRefSeg2 Frozen | none | 37.74 | 78.12 | 92.07 | .569 | -1.012 | 8.41 | 30.89 | 60.20m |
| RSRefSeg2 Prompter | 82.77M | 50.96 | 92.07 | 98.80 | .695 | -.157 | 24.52 | **44.59** | **48.35m** |
| RSRefSeg2 Vision-LoRA | 86.97M | **51.56** | **92.19** | **98.92** | **.697** | -.163 | **24.64** | 43.63 | 49.85m |
| GT reference mask oracle | privileged | 89.06 | 98.80 | 100.00 | .931 | 5.145 | 18.15* | 41.11* | 50.55m* |

### val_unseen

| Method | Trainable | Top-1 | Top-4 | Top-8 | MRR | Margin | R@20m | R@40m | Median dist |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current SigLIP | none | 32.88 | 64.65 | 81.93 | .498 | -.015 | 18.18 | 35.24 | 58.30m |
| Whole RGB SigLIP2 | none | 38.95 | 72.73 | **85.30** | .565 | -.008 | 25.25 | 45.34 | 44.31m |
| Region-aware SigLIP2 | none | 27.05 | 57.58 | 72.39 | .442 | -.017 | 23.12 | 42.99 | 50.85m |
| RSRefSeg2 Frozen | none | 32.21 | **79.01** | **91.69** | .542 | -1.101 | 23.01 | 45.68 | 45.47m |
| RSRefSeg2 Prompter | 82.77M | 48.60 | 74.97 | 83.16 | .624 | -.324 | **28.17** | **52.75** | **36.68m** |
| RSRefSeg2 Vision-LoRA | 86.97M | **49.16** | 73.74 | 82.72 | **.625** | -.334 | 27.72 | 51.40 | 38.35m |
| GT reference mask oracle | privileged | 73.29 | 99.55 | 100.00 | .855 | 7.963 | 31.87* | 59.26* | 35.03m* |

`*` The mask oracle is meaningful for candidate identity/rank. Its uniform
mask has no unique peak, so argmax-based localization distance is not an oracle
distance and must not be interpreted as one.

The historical 43–45% scene-evidence numbers used same-map pair accuracy and
are not silently mixed with this candidate-ranking benchmark. The recomputed
baselines above use exactly the same 832/891 frames and candidate protocol as
RSRefSeg2.

## 4. Experiment A–D decisions

- **A — Frozen:** worse than SigLIP2 on unseen Top-1 (-6.74 points).
  Remote-sensing pretraining alone does not transfer.
- **B — Prompter-only:** gains +10.94 seen and +9.65 unseen Top-1 over SigLIP2
  dense. Adaptation is necessary.
- **C — Vision-LoRA:** gains +11.54 seen and +10.21 unseen over SigLIP2 dense,
  but only +0.60/+0.56 over B. LoRA's marginal value is negligible.
- **D — SAM:** skipped. C did not clear the robust improvement gate, and the GT
  crop test proves that boundary refinement cannot resolve candidate identity.

Neither B nor C showed sustained monotonic val_seen improvement; both oscillate
across epochs. Training therefore stopped at the pre-registered 5 epochs rather
than mechanically extending to 10.

## 5. GT crop observability diagnostic

GT boxes are used only to create diagnostic crops. These results are not a
deployable inference path.

| Method | val_seen Top-1 | val_seen Top-4 | val_unseen Top-1 | val_unseen Top-4 | val_unseen R@20m |
|---|---:|---:|---:|---:|---:|
| Current SigLIP crop | 29.69 | 66.95 | 26.49 | 59.60 | 22.11 |
| SigLIP2 crop | 44.71 | 69.11 | 27.05 | 57.58 | 23.12 |
| RSRefSeg2 Frozen crop | **69.35** | **90.38** | **47.36** | **83.73** | **48.04** |
| RSRefSeg2 Prompter crop | 48.44 | 81.97 | 42.65 | 77.22 | 43.32 |
| RSRefSeg2 Vision-LoRA crop | 51.32 | 83.77 | 43.55 | 78.68 | 44.56 |

The expected “GT crop reaches 70–80% Top-1” condition is decisively false.
Correct crops remain ambiguous in unseen cities, and CityNav fine-tuning
damages crop identity retrieval. This is direct evidence for an appearance/observability
bottleneck rather than a pure proposal/localization bottleneck.

## 6. Language ablation

### Prompter-only

| Input text | val_seen Top-1 | val_seen Top-4 | val_unseen Top-1 | val_unseen Top-4 |
|---|---:|---:|---:|---:|
| Full instruction | 50.96 | 92.07 | 48.60 | 74.97 |
| Referenced phrase | 50.96 | 93.27 | 50.39 | 79.57 |
| Name/category only | **52.76** | **93.99** | **51.85** | **81.71** |
| Full without spatial relation | 51.08 | 93.63 | 49.94 | 75.65 |

Full language helps on seen data but hurts unseen relative to name/category.
Removing spatial words changes little. The model is primarily doing landmark
retrieval, not robust relation/context reasoning. Freezing the text encoder was
therefore the correct first-stage choice; the evidence does not justify text
fine-tuning.

## 7. Category, distance, size, and distractors

Only three annotated referenced categories occur in this visible benchmark:
Building, TrafficRoad, and three Footpath samples. The requested water,
vegetation, bridge, parking, and sports-field category claims cannot be made
from this subset.

Prompter-only `val_unseen` Top-1:

| Slice | Samples | Top-1 | R@20m |
|---|---:|---:|---:|
| Building | 357 | 54.06 | 22.41 |
| TrafficRoad | 531 | 44.63 | 31.64 |
| visible area: small | 46 | 43.48 | 26.09 |
| visible area: medium | 209 | 28.71 | 23.92 |
| visible area: large | 636 | 55.50 | 29.72 |
| distance 0–20m | 190 | 62.63 | 53.68 |
| distance 20–40m | 242 | 57.44 | 33.88 |
| distance 40–80m | 265 | 45.28 | 14.34 |
| distance >80m | 194 | 28.35 | 14.95 |
| 1–4 candidates | 376 | 77.66 | 44.95 |
| 5–8 candidates | 105 | 40.95 | 22.86 |
| 9+ candidates | 410 | 23.90 | 14.15 |

Distance and same-scene candidate density dominate. Top-1 drops from 77.4% to
23.9% when moving from at most four candidates to nine or more, and to 28.4%
beyond 80m. This is exactly the setting in which merely sharpening a mask is
unlikely to help.

## 8. Runtime and memory

| Run | Time | Peak GPU | Trainable parameters |
|---|---:|---:|---:|
| Frozen inference, both splits | 58.9s | 5.21 GiB | 0 |
| Prompter-only, 5 epochs | 371.5s | 9.87 GiB | 82.77M |
| Vision-LoRA, 5 epochs | 505.0s | 13.75 GiB | 86.97M |
| Prompter eval + 4 language modes | 186.9s | 5.21 GiB | 0 |
| Each RSRefSeg2 GT-crop diagnostic | 292–296s | 8.10 GiB | 0 |

Checkpoints and full run provenance live under
`/mnt/windows-data/hett-rsrefseg2/runs`. The 80 requested qualitative panels
(20 success, 20 failure, 20 frozen-to-LoRA improvement, 20 LoRA-still-failed)
are under `/mnt/windows-data/hett-rsrefseg2/artifacts/qualitative_fixedpad`.

## 9. Static B0 context

Static B0 is a final-goal localization benchmark, while this experiment ranks
visible referenced landmarks. Their Top-1 values are not the same task and are
not merged. For context, the prior frozen result on `val_unseen` is R@1/20
21.43%, R@4/20 46.98%, R@8/20 65.96%, and R@16/20 80.57%.

No B0 update was performed because the grounding model did not pass the robust
integration gate. The privileged “B0 + perfect referenced geometry” number
(69.74% R@1/20) remains an upper bound, not a fair RSRefSeg2 baseline.

## 10. Final diagnosis and recommendation

1. **Visual localization:** real, secondary problem. Prompter adaptation gives
   a substantial full-image gain.
2. **Visual observability/identity:** primary problem. About 20% of unseen
   frames do not contain the reference, and GT-crop Top-1 is only 47.36%.
3. **Language grounding:** also weak. Complex full instructions do not
   generalize better than a name/category query.
4. **Overall:** a mixed problem, dominated by top-down appearance ambiguity,
   distance, and same-class distractors.

**Do not integrate RSRefSeg2 into HETT yet.** Keep this branch as a diagnostic
and hard-negative/localization pretraining tool. The next justified experiment
is genuine first-person/oblique multi-view data plus landmark appearance
memory, with the top-down geometry stream retained separately. Revisit the
Prompter only if GT-crop retrieval on those views first exceeds roughly
70% unseen Top-1. Do not fine-tune SAM or the text encoder under the current
camera/data interface.
