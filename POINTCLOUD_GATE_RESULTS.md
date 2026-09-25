# Point-cloud perspective supervision: first validation

## Protocol

- Renderer and models are independent of GeoNav and HETT.
- Frames are selected from recorded poses only; target coordinates and target
  identity are not read for selection or rendering.
- Missing point-cloud pixels use label 255 and are ignored in loss and metrics.
- Training uses 569 keyframes from 24 `train_seen` maps.
- Checkpoint selection uses 529 keyframes from 23 `val_seen` maps.
- Final perspective evaluation uses 96 keyframes from all four `val_unseen` maps.
- Mixed training replays 576 train-seen orthomosaic patches to prevent domain
  forgetting.

Mean raw valid-pixel fractions are 50.18% for train_seen, 50.68% for val_seen,
and 54.22% for val_unseen. The val_unseen 10th percentile is 25.08%, so invalid
pixels must remain explicitly masked.

## Perspective semantic result

| val_unseen metric | orthomosaic-only model | mixed point-cloud model | delta |
|---|---:|---:|---:|
| Foreground mIoU | 21.18% | **42.84%** | **+21.66** |
| Building IoU | 35.96% | **58.40%** | +22.43 |
| Car IoU | 19.31% | **46.53%** | +27.22 |
| Ground IoU | 21.78% | **33.26%** | +11.47 |
| Parking IoU | 7.67% | **33.17%** | +25.50 |

## Orthomosaic retention

| val_unseen metric | original model | mixed point-cloud model | delta |
|---|---:|---:|---:|
| Foreground mIoU | 53.65% | 53.13% | -0.52 |
| Building IoU | 64.03% | **65.51%** | +1.48 |
| Car IoU | 66.36% | **66.49%** | +0.14 |

Point-cloud-only fine-tuning caused severe orthomosaic forgetting. Balanced
replay removed nearly all aggregate loss while keeping the perspective gain.

## Existing top-down named-landmark gate

| val_unseen metric | original model | mixed point-cloud model |
|---|---:|---:|
| End-to-end named Top-1 | 62.56% | **72.41%** |
| Named Top-1 given detection | 70.95% | **78.19%** |
| Median projected component error | 16.33 m | **8.42 m** |
| Wrong-pose Top-1 | 15.08% | 14.89% |
| Wrong-name Top-1 | 11.17% | 7.45% |

The controls do not improve with the main score, which argues against a map or
name shortcut.

## Direct perspective named-landmark gate

Only instruction landmarks with at least eight valid building points are
evaluated. There are 42 such queries in the 96-frame sample.

| val_unseen metric | original model | mixed point-cloud model |
|---|---:|---:|
| Landmark detection at >=25% coverage | 61.90% | **97.62%** |
| Named Top-1 given detection | **73.08%** | 53.66% |
| End-to-end named Top-1 | 45.24% | **52.38%** |

The lower conditional Top-1 is not a contradiction: the adapted model detects
almost every visible landmark, including difficult adjacent buildings that are
merged into large components. End-to-end identity still improves by 7.14 points.
The next bottleneck is component-to-polygon binding, not building visibility.

## Oracle ladder and RGB depth test

The component bottleneck was isolated by replacing whole-component identity
with per-pixel polygon binding on the same 42 visible val_unseen queries.

| Binding method | End-to-end confirmation |
|---|---:|
| Teacher semantic + teacher depth + landmark ID | 100.00% |
| Predicted semantic + teacher depth, per pixel | **97.62%** |
| Predicted semantic + teacher depth, whole component | 52.38% |
| Predicted semantic + RGB-predicted depth, per pixel | 42.86% |

The RGB semantic/depth model was selected on val_seen at epoch 13. On
val_unseen it obtains 44.43% perspective foreground mIoU, but absolute depth
MAE is 36.50 m. Expanding the accepted landmark polygon to 20 m raises the
correct-name result to 71.43%, but also raises the nearest-wrong-name control to
66.67%. Therefore spatial tolerance cannot repair generic monocular depth
without destroying identity discrimination.

## Direct map-conditioned landmark projection

CityRefer already provides each named building's footprint, center height, and
dimensions. The independent binder now projects that annotated 3D prism into
the camera using the measured pose, then intersects it with the RGB-only
building prediction. It does not predict absolute depth and is not trained on
val data.

| Split | Visible queries | Correct name + pose | Wrong pose | Nearest wrong name |
|---|---:|---:|---:|---:|
| val_seen | 253 | **93.28%** | 4.35% | 11.51% |
| val_unseen | 42 | **95.24%** | 0.00% | 9.52% |

With a two-pixel calibration tolerance, val_unseen reaches 97.62%, while the
wrong-pose and wrong-name controls remain 2.38% and 16.67%. Larger dilation is
not used because it increases wrong-name acceptance. These metrics use teacher
semantic/depth only to determine visibility and score the result; runtime inputs
are rendered RGB, camera pose, and the named map landmark prism.

This is a feasibility result on point-cloud-rendered RGB, not yet evidence of
transfer to simulator RGB or of a navigation SR/SPL improvement.

## Orthomosaic RGB transfer gate

The AirSim endpoint was unavailable, so simulator perspective transfer was not
claimed. As an intermediate domain gate, the same named-polygon binding was
evaluated on the existing independent altitude benchmark: 1,910 val_unseen
orthomosaic RGB crops and 1,243 evaluable named-landmark queries. Goal target
coordinates are not used.

| Semantic checkpoint | Correct binding | Wrong pose | Wrong-name leakage |
|---|---:|---:|---:|
| Orthomosaic-only baseline | 79.32% | 0.72% | 3.22% |
| **Mixed point-cloud + orthomosaic** | **85.44%** | 0.88% | 3.22% |
| Joint semantic + absolute depth | 75.30% | 0.48% | 3.22% |

Mixed point-cloud supervision improves the real-RGB gate by 6.11 percentage
points over the orthomosaic-only baseline. The absolute-depth multitask model
hurts this gate and is not selected.

For the selected mixed model, val_seen is 93.98% and val_unseen is 85.44%, an
8.54-point generalization gap. The val_unseen altitude breakdown is 92.59% at
30 m, 89.47% at 50 m, 86.38% at 60 m, 87.37% at 80 m, and 77.27% at 120 m.
The high-altitude decline is caused primarily by small building footprints, so
coarse navigation should preserve the map geometry prior rather than expecting
RGB semantics alone to remain equally reliable.

This gate uses real orthomosaic RGB, not perspective AirSim RGB. AirSim is an
optional robustness domain, not a CityNav data source or a prerequisite for the
standard benchmark. CityNav uses real-city orthomosaics and SensatUrban
geometry; human demonstrations were collected in a web 3D environment. The
standard local evaluation path uses the crop client, so temporal validation
continues on CityNav orthomosaic RGB.

## Fixed-50-m temporal landmark belief

A positive-only noisy-OR updater accumulates landmark evidence and never treats
an unconfirmed frame as negative evidence, because occlusion is unknown. The
confirmation threshold is selected on val_seen only.

| val_unseen metric | First observation | Temporal (2.52 observations) |
|---|---:|---:|
| Named-landmark confirmation | 78.57% | **95.92%** |

Temporal accumulation improves confirmation by 17.35 points. Wrong-pose and
nearest-wrong-name acceptance remain 1.02% and 6.12%. This proves persistent
landmark identity, not target localization.

## Fixed-50-m temporal target-candidate replay

The independent geometry generator's five candidates are projected into each
RGB crop. A train_seen/val_seen text probe predicts the instruction target type;
the RGB semantic model supplies type evidence around each candidate. The fusion
weight is selected on val_seen only. The predeclared 10-m candidate neighborhood
is retained for the primary result.

| val_unseen replay method | Hit@20 | mean NE |
|---|---:|---:|
| Geometry generator Top-1 | 21.88% | 43.36 m |
| Single-frame visual fusion | 16.41% | 44.54 m |
| **Multi-frame visual fusion** | **28.91%** | **38.39 m** |
| Wrong target type | 20.31% | 42.08 m |
| Evidence shuffled across episodes | 20.31% | 44.13 m |
| Top-5 oracle | 51.56% | 27.02 m |

Multi-frame RGB improves Hit@20 by 7.03 points over the generator's first
candidate, while single-frame fusion hurts. Radius sensitivity is directionally
consistent: 5, 10, 15, and 20 m produce 27.34%, 28.91%, 29.69%, and 31.25%
Hit@20. These radii were inspected after the primary 10-m result, so the larger
numbers are robustness analysis and are not used for model selection.

This is a 128-episode teacher-forced replay. The benchmark observation poses
were selected from human trajectories by target-distance bands; therefore it is
not an autonomous navigation or SR/SPL result. It establishes that correctly
aligned, correctly typed multi-frame RGB evidence can close part of the gap to
the Top-5 ceiling. The next fair test must generate observation poses without
reading target distance.

## Decision

Keep mixed semantic training, but do not use generic monocular absolute depth as
the primary binding path. Use direct map-conditioned projection as the geometric
prior and multi-frame RGB as evidence. The next gate is target-free next-view
selection at fixed 50 m, followed by closed-loop SR/SPL. AirSim is optional.
