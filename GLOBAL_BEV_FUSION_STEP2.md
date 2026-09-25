# Recurrent global-BEV fusion — Online Todo 2

## Scope

This is a structural overfit gate, not a navigation result.  A lightweight
ConvGRU-style model maintains a 64x64 world-aligned BEV.  Current RGB is
projected into the BEV by a fixed pose transform, and the recurrent state is
updated only inside the current frustum.  The model predicts a coarse cell and
a continuous within-cell offset, supervised by the exact target coordinate.
The 20 m radius is used only for evaluation.

The learned model does not receive raw pose or target coordinates.  Pose is
used only by the fixed RGB-to-BEV projector.  Labels are joined only after all
inference inputs have been materialized.

## Gate result

| Split | Count | Mean NE | Hit@5 | Hit@20 |
|---|---:|---:|---:|---:|
| train_seen overfit | 8 | 14.07 m | 87.5% | 87.5% |
| val_seen diagnostic | 8 | 52.29 m | 0.0% | 12.5% |

The predeclared overfit gate required at least 90% train Hit@20 and at most
10 m train mean NE, so the step failed.

Seven training episodes were fitted to 0.03--0.13 m.  The remaining episode
was 112.07 m wrong.  Its true target was outside the pose-conditioned frustum
in all six collected frames, while the rollout followed a predicted Top-1
candidate 63.7 m from the target.  The model therefore never received current
visual evidence at the true location.

RGB is connected to the model: zeroing it changes final logits by 0.105 mean
absolute value, and shuffling it changes logits by 0.107.  Changing the hidden
target labels after forward changes predictions by exactly zero.  Unit tests
also confirm that recurrent state is bitwise unchanged outside visibility.

## Conclusion

The BEV recurrence, exact coordinate offset and leakage boundary work, but the
current target-free Top-1 rollout does not guarantee that the training sequence
ever observes the true target.  A larger model cannot repair missing evidence.
Before training the exploration-value head, the next experiment should provide
target-free multi-hypothesis observation coverage and measure whether it makes
the true region observable; target coordinates may score coverage afterward
but must not select the route.

Machine-readable output: `runs/global_bev_fusion_pilot/results.json`.
