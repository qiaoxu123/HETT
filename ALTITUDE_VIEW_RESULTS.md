# Active-altitude visual validation results

Date: 2026-09-23.  Development splits only; `test_unseen` was not loaded.

## Experiment 1: geometry and image information

Run: `runs/altitude_view_geometry128_s0` (ignored runtime artifact).

- Map-balanced seed-0 sample: 128 episodes from each of `val_seen` and
  `val_unseen`.
- 3,790 real altitude views plus matched digital-zoom controls, 512 by 512.
- Poses come from the human path near 80, 40 and 20 m target-distance bands.

At the 20 m band:

| Split | Height | N | target visible | fully visible | mean target pixels | any landmark visible |
|---|---:|---:|---:|---:|---:|---:|
| val_seen | 30 m | 126 | 100.0% | 78.6% | 14,913 | 90.5% |
| val_seen | 50 m | 126 | 100.0% | 97.6% | 6,293 | 97.6% |
| val_seen | 60 m | 126 | 100.0% | 99.2% | 4,480 | 98.4% |
| val_seen | 80 m | 126 | 100.0% | 100.0% | 2,560 | 99.2% |
| val_seen | 120 m | 126 | 100.0% | 100.0% | 1,154 | 100.0% |
| val_unseen | 30 m | 126 | 100.0% | 69.0% | 25,539 | 84.1% |
| val_unseen | 50 m | 126 | 100.0% | 90.5% | 13,610 | 94.4% |
| val_unseen | 60 m | 126 | 100.0% | 93.7% | 10,279 | 97.6% |
| val_unseen | 80 m | 126 | 100.0% | 97.6% | 6,191 | 99.2% |
| val_unseen | 120 m | 126 | 100.0% | 100.0% | 2,789 | 100.0% |

At the 80 m band, 30 m height sees the target in only 4.7% of val_seen and
3.1% of val_unseen cases; 120 m sees it in 100% of both.  A globally low
camera is therefore invalid for search.

At 20 m distance, a real 60 m image has 6.03x (`val_seen`) and 6.31x
(`val_unseen`) the Laplacian variance of a digital zoom made from the matched
120 m image.  For 30 m the ratios are 25.96x and 27.31x.  Lower crops contain
substantially more orthomosaic detail rather than merely larger objects.

Interpretation: use 120/80 m for search, descend first to about 60 m for local
grounding, and permit 30 m only after the target belief is centered and a
field-of-view containment check passes.

## Experiment 2: frozen semantic grounding

The formal exact-common-visibility set has 195 poses and 1,755 real/digital-
zoom images.  Eighteen instructions are truncated to the SigLIP2 64-token
limit.  Official RefSegRS checkpoint integrity:

`6018af204120cf14e7555ff067f6fa4a921eba084c33be308bc65bc540f826ee`

Environment validation history is preserved in the failed smoke run folders:

- r1: missing `hydra-core`;
- r2: missing native SAM2 `.pt` cache;
- r3: missing `iopath`;
- r4: missing Hugging Face `datasets`;
- r5: complete, 9/9 images evaluated.

The successful zero-shot smoke has Box Acc@0.25 = 0/9 and very low absolute
IoU, consistent with the earlier observation that unadapted RefSegRS transfers
poorly to CityRefer.  The smoke is an environment check, not evidence about
height.  A full exact-paired run is required before deciding whether to repeat
the train_seen-only CityRefer fine-tuning.

Batch 32 is invalid even on a 98 GB GPU because the intermediate tensor
`[32, 1152, 256, 256]` exceeds the `upsample_bilinear2d_nhwc` INT_MAX element
limit.  Batch 16 is the maximum validated batch: about 2.5 seconds per batch,
100% sampled GPU utilization and roughly 29 GB recorded peak memory.
