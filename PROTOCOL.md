# Relational landmark heatmap protocol

## Question

Can named landmark contours and instruction geometry produce a useful target heatmap
before visual/action reasoning?

## Frozen scope

- Train on `train_seen`; evaluate all episodes in `val_seen` and `val_unseen`.
- Do not use RGB, target contours, target attributes, validation labels, or navigation
  rollouts as model inputs.
- Each named landmark keeps its own contour and identity.  The instruction is encoded
  once per landmark after marking that landmark as `<ref>` and other named landmarks
  as `<other>`.
- Fixed geometric bases: inside, near, ring, four cardinal half-planes, and four
  diagonal sectors.  A learned language head mixes these bases; a small CNN refines
  the combined field.  A separate pair channel represents the region between
  landmarks.
- Output is a global 64 x 64 probability field.  Training target is a 10 m Gaussian.
- Top-K uses metric non-maximum suppression with a 20 m separation radius.

## Metrics

- Top-1 Hit@20 and Top-5 Recall@20.
- Median/P90 Top-1 error and probability mass inside 20 m.
- The same metrics by relation family: left/right, front/behind, near, between,
  cardinal, ordinal, and other.
- Compare with first-landmark centroid and an untrained contour-distance heatmap.

## Decision set before running

The first-stage design passes this diagnostic only if:

1. `val_unseen` Top-5 Recall@20 is at least 70%;
2. `val_unseen` Top-1 Hit@20 is at least 35%; and
3. `val_seen` Top-1 does not trail `val_unseen` by more than 15 percentage points.

This is a target-proposal diagnostic.  It is not navigation SR and it does not use
`test_unseen`.
