# Multi-hypothesis candidate generation protocol

## Question

Can five explicitly diverse heatmap heads improve target proposal recall over taking
five NMS peaks from one heatmap?

## Fixed comparison

- Data: train only on `train_seen`; select/check on `val_seen`; report final development
  result on `val_unseen`. Do not run `test_unseen` during development.
- Inputs: instruction text, resolved named-landmark identities and contours, and the
  landmark-pair field. No RGB, target contour, target identity, navigation trajectory,
  or validation label is an input.
- Supervision: the target coordinate is used only to form the train-set Gaussian and
  20 m loss mask. It is not passed to `forward` or candidate extraction.
- Control: the committed factorized single-head model from experiment 28, six epochs,
  seeds 2701/2702/2703.
- Variant: the same trunk with five output heads. Training normalizes probability over
  head and pixel jointly, optimizes spatial target likelihood and mass inside 20 m, and
  adds a small pairwise head-overlap penalty. Inference takes at most one peak from each
  head and fills duplicate modes from the summed map without target access.

## Metrics and gate

Primary: Top-5 Recall@20 on `val_unseen`. Secondary: Top-1 Hit@20, median/P90 error,
and results by landmark count/relation family.

Promote only if the three-seed mean improves the experiment-28 `val_unseen` Top-5
baseline of 57.20% by at least 3 absolute points without reducing Top-1 by more than
2 points. The longer-term candidate-generation target is at least 80% Top-5 Recall@20.

## Staged execution

1. CPU smoke test and leakage/shape tests.
2. One full seed after the existing HETT GPU job releases the device.
3. Only if seed 2701 improves, run seeds 2702 and 2703.
