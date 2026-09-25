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

## Decision

The point-cloud data passes the first feasibility gate. Keep mixed semantic
training. The next experiment should train `MapLandmarkBinder` on projected
component/polygon overlap and compare it with the current hand-written binding.
Do not claim a navigation SR/SPL gain until the resulting evidence is inserted
into the temporal belief map and tested in closed loop.
