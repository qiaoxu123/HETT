# Target-only HETT keyframe experiment

## Question

Can HETT's language, RGB and map encoders learn a more accurate target
position when supervision is concentrated on informative images rather than
accumulated along a complete action trajectory?

## Isolated change

- Keep the released HETT language, RGB, map and multimodal Transformer path.
- Keep up to nine named landmarks independently.
- Add each landmark's 16-point normalized contour; candidate geometry uses
  both centroid-relative and nearest-boundary-relative features.
- Process one selected image per episode and supervise only the target belief.
- Disable direction, progress, action, stopping and rollout losses.
- The output used by this experiment is the target BEV distribution and its
  world-coordinate peak.
- The official HETT checkpoint initializes shared encoders with `--warm_start`;
  its stored epoch is not treated as the new experiment's resume epoch.

## Keyframe rule and leakage boundary

The primary train/evaluation view is selected from the human trajectory in a
25--60 m annulus around the target.  This is deliberately outside the 20 m
success circle.  If the annulus is empty, select the closest available frame
that remains at least 25 m away; only trajectories wholly inside that radius
fall back to their farthest frame.  Episodes whose entire recorded trajectory
remains inside 25 m are masked out of the primary loss and metrics rather than
being counted as easy target-neighbourhood predictions.

Target truth may select a **training/evaluation diagnostic view** and construct
the target label, but it is never passed to the model.  Because view selection
is privileged, this experiment measures a perception/localisation upper bound,
not a deployable navigation score.

## Loss

The only optimized quantities are:

1. exact 41x41 target-cell cross entropy;
2. negative log probability mass inside the metric 20 m target region;
3. within-cell offset SmoothL1.

Loss is normalized per active sample, not accumulated over trajectory length.

## Metrics

Report target error mean/median, Hit@20 and Hit@50 separately for `val_seen`
and frozen `val_unseen`.  Also report mean camera-to-target distance and the
fraction accidentally inside 20 m (expected to be zero for the main annulus).

Semantic-reliance controls use the same checkpoint and views:

- normal condition;
- same-map wrong instruction plus wrong named-landmark set;
- empty instruction;
- zero RGB;
- pose only (empty instruction, zero RGB, no valid landmarks).

A high unseen Hit@20 is evidence of semantic understanding only when the
same-map wrong-condition and pose-only controls decrease materially.

`test_unseen` remains disabled unless `--include_test_unseen` is explicit.
