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

Following the requested perception-upper-bound experiment, the primary
train/evaluation image is rendered at the **final recorded human-trajectory
pose**.  No earlier trajectory frame or accumulated trajectory loss is used.

Target truth constructs the target label but is never passed to the model.
The human endpoint is privileged and is often already close to the target, so
this experiment measures a perception/localisation upper bound, not deployable
navigation.  Every result must therefore include the `pose_copy_hit20`
baseline: predicting the camera XY without reading RGB or language.

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

Before full training, `--target_eval_train_split` is used with a tiny fixed
subset to verify that the new head can overfit its target labels.  This
diagnostic number is never reported as generalization performance.

`test_unseen` remains disabled unless `--include_test_unseen` is explicit.
