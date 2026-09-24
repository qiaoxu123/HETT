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

The shared GPU is occupied by the independent three-epoch HETT training job.
Per repository policy, the full seed is not launched concurrently. Once the
device is released, run seed 2701 for six epochs. Continue to the other two
seeds only if `val_unseen` Top-5 improves over 57.20%.
