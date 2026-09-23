# Active-altitude visual validation protocol

## Question

Before changing HETT navigation training, test whether lower-altitude imagery
actually improves language-conditioned target grounding.  Height is the only
camera variable in a pair: instruction, horizontal position, yaw, map and
target are identical.

This is an orthomosaic multi-scale experiment.  It measures the benefit of
sampling more source pixels and changing field of view, but cannot demonstrate
new 3-D viewpoint, facade visibility or occlusion effects.

## Data discipline

- Development uses `val_seen` and `val_unseen` only.  `test_unseen` is excluded.
- Episodes are sampled map-balanced with seed 0.
- Poses are selected from the human path near 80 m, 40 m and 20 m horizontal
  target-distance bands.
- Altitudes are 30, 50, 60, 80 and 120 m AGL.  Images and target masks are
  rendered at 512 by 512.
- The strict semantic subset contains only poses where the complete target is
  visible at every tested altitude.  This prevents visibility changes from
  masquerading as recognition changes.
- For every low-altitude image, a digital-zoom control is made by center
  cropping the corresponding 120 m image and resizing it.  Real-low versus
  digital-zoom therefore isolates newly sampled image detail.

## Ordered experiments

1. Geometry and image-information audit: target visibility, target pixels,
   landmark co-visibility, real-low/digital-zoom PSNR and edge-detail variance.
2. Frozen official RefSegRS checkpoint on the strict paired subset.  Primary
   metrics are box IoU, Box Acc@0.25/0.50 and center hit; footprint mask IoU is
   secondary.
3. If the official model is too weak on CityRefer, reproduce the prior
   train_seen-only two-epoch CityRefer fine-tune, select on validation data,
   then repeat exactly the same frozen paired test.
4. Only if lower views beat both 120 m and digital zoom on `val_unseen` without
   excessive target/landmark loss, connect the altitude controller to HETT and
   run navigation ablations.

## Pilot-32 result (seed 0)

The map-balanced pilot contains 32 episodes per split and 950 real views.
At the 20 m distance band:

| Split | Altitude | Fully visible | Mean target pixels |
|---|---:|---:|---:|
| val_seen | 30 m | 74.2% | 18,697 |
| val_seen | 60 m | 96.8% | 6,564 |
| val_seen | 120 m | 100.0% | 1,710 |
| val_unseen | 30 m | 68.8% | 17,852 |
| val_unseen | 60 m | 100.0% | 5,570 |
| val_unseen | 120 m | 100.0% | 1,423 |

At 80 m distance, targets are almost never visible at 30 m altitude and are
fully visible in roughly 97% of 120 m views.  Thus the data supports a staged
policy: high altitude for search, about 60 m for approach/local grounding, and
30 m only for a final refinement observation after centering the target belief.

Real 30 m crops have about 27 times the Laplacian variance of digital zooms
from 120 m in both splits.  This proves that descent exposes substantially more
orthomosaic detail; it does not yet prove that the extra detail improves
language grounding.  Experiment 2 answers that next.

The strict pilot semantic set has 48 common-visibility poses and 432 real/zoom
images.  It is intentionally small for environment and checkpoint validation;
the formal run should increase the map-balanced sample count after the pipeline
passes end to end.
