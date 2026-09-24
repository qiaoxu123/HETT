# Multi-hypothesis candidate generation progress

## Baseline evidence

- Experiment 28 single factorized heatmap, three-seed `val_unseen` mean:
  Top-1 Hit@20 27.81%, Top-5 Recall@20 57.20%.
- The deterministic direction-by-distance candidate bank has a privileged
  `val_unseen` Hit@20 upper bound of 91.32%. This coordinate is used only for
  that diagnostic and training supervision, never as inference input.
- The later frozen five-candidate set reaches 74.23% Top-5 Hit@20 on
  `test_unseen`, but that split is not used to develop this experiment.

## Implemented variant

The single spatial output is replaced by five heatmap heads. Probability is
normalized jointly over head and pixel; the training loss uses the summed
spatial probability, 20 m probability mass, and a head-overlap penalty.
Inference takes one peak per distinct head and fills duplicates by target-free
NMS over the summed map.

## Completed checks

- Four unit tests pass, including output shapes, invalid-landmark masking and
  an explicit signature check that candidate extraction cannot receive target
  coordinates.
- CPU smoke run: 64 train / 32+32 validation samples, one epoch, seed 2701.
  The full train/evaluate/report/checkpoint path completed in 105.5 seconds.
- Smoke accuracy is not evidence because the subset and budget are deliberately
  tiny. The observed head cosine overlap was 0.996, so head collapse must be
  checked in the full run and the diversity term may need strengthening.

## Current blocker and next execution

The user authorized a parallel full seed on the 96 GB RTX PRO 6000. Seed 2701
completed six epochs with batch size 16:

| Split | Top-1 Hit@20 | Top-5 Recall@20 | Median error |
|---|---:|---:|---:|
| val_seen | 39.35% | 57.61% | 25.38 m |
| val_unseen | 25.62% | 41.05% | 37.04 m |

Against experiment 28, unseen Top-1 fell from 27.81% to 25.62% and Top-5 fell
from 57.20% to 41.05%. The head-overlap diagnostic did improve from 0.590 after
epoch 1 to 0.100 after epoch 6, so the heads separated, but separation forced
probability into distinct wrong modes rather than improving target coverage.
The pre-registered single-seed continuation gate failed; seeds 2702/2703 will
not be run.

Resource observation: the variant used only about 0.93 GiB additional GPU
memory and stayed at 80 C or below, but concurrent HETT steps slowed from about
9 seconds to about 15 seconds. Accuracy remains interpretable; runtime is not
comparable to an isolated run.
