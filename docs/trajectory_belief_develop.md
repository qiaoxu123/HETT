# Semantic Core-Anchor Trajectory Navigation

The develop branch now separates **final-goal localization** from **next
trajectory decision**.

The 7x7 final-goal heatmap remains a long-range spatial prior.  Trajectory
generation is no longer driven directly by all 49 goal cells.  Instead the
policy predicts the next sparse human-intent **core anchor**, then generates a
short receding-horizon trajectory toward that anchor.

## Human core anchors

CityFlight keyboard traces contain dense low-level samples.  Training derives
sparse anchors from four sources:

1. **route turns**: RDP corners preserve meaningful XY route changes;
2. **view changes**: large first-person yaw changes are retained;
3. **referenced landmark passages**: the closest demonstrated point to each
   instruction-referenced landmark is retained when the route passes within a
   configurable radius;
4. **final goal**: the demonstrated endpoint is always retained.

Small spatial/yaw jitter is removed before the core set is formed.

Default extraction parameters:

```
human_anchor_min_step_m = 1.0
human_anchor_rdp_tolerance_m = 2.5
human_anchor_yaw_keyframe_deg = 30
human_anchor_landmark_radius_m = 30
human_anchor_min_lookahead_m = 8
```

At each teacher/student training state the next future core anchor is selected.
For teacher states the known human arc coordinate is used directly; for
student states the current XY is projected continuously onto the human path.

## Semantic anchor inference

Inference does **not** use the human trajectory.  The next core-anchor belief
is predicted from information already available to the agent:

```
final-goal 7x7 heatmap
instruction representation
current visual state
referenced landmark names (BERT)
referenced landmark coordinates
map/history-conditioned HETT candidate features
```

Referenced landmark names reuse contextual token features from the single
trainable instruction-BERT forward.  No extra landmark BERT pass is required.
Each contextual name embedding is paired with its geographic coordinate.
Candidate-to-landmark attention combines semantic similarity with geographic
proximity.

The anchor branch uses the final-goal heatmap logits as the initialization
prior and learns a residual core-anchor score.  The residual head is
zero-initialized, so the first forward pass preserves the previous develop
trajectory ranking.  Landmark, visual and language contexts enter through
separate learnable residual gates.

## Two spatial beliefs with different roles

```
Final-goal heatmap:
    Where is the destination roughly?

Semantic core-anchor belief:
    What is the next meaningful decision point on the way there?
```

These are intentionally not the same target.

## Anchor-conditioned trajectory

For each 7x7 anchor region the model predicts:

```
continuous anchor offset:  dx, dy
trajectory residual:       5 x (dx, dy)
```

The selected anchor is refined continuously inside its coarse cell.  A straight
fixed-horizon path is then constructed toward it:

```
10 m / 25 m / 50 m / 100 m / core anchor
```

Horizons beyond the human core anchor are masked in the trajectory loss.  This
prevents the trajectory head from being supervised on motion that belongs to a
later semantic decision segment.

The residual head is zero-initialized, so early training starts from stable
straight anchor-directed motion.

## Training objectives

```
L =
  L_HETT
+ lambda_heatmap * L_final_goal_heatmap
+ lambda_anchor * L_core_anchor_region
+ lambda_anchor_pos * L_core_anchor_position
+ lambda_traj * L_anchor_conditioned_trajectory
```

The existing final-goal coordinate and progress losses are preserved for
diagnostics/backward compatibility.

Default new weights:

```
semantic_anchor_loss_weight = 0.5
semantic_anchor_position_loss_weight = 1.0
trajectory_loss_weight = 1.0
```

## Student execution

```
Language + RGB + map/history
            |
           HETT
            |
      final-goal heatmap
            |
    semantic core anchors
            |
       NMS + Top-K
            |
  continuous anchor refinement
            |
anchor-conditioned trajectories
            |
execute first ~10 m waypoint
            |
       observe + replan
```

The learned student policy remains trajectory-only; Stage-1/Stage-2 execution
is not used.

## Teacher rollout

Teacher and student use the same network and losses.  They differ only in
next-state generation:

```
teacher -> recorded human path + teacher_step_m
student -> predicted core-anchor trajectory first waypoint
```

Teacher x/y/z/yaw are interpolated from the human demonstration.

## Diagnostics

`navigation_diagnostics.jsonl` continues to store final-goal heatmap data and
now also reports semantic-anchor quality:

```
semantic_anchor_top1_acc
semantic_anchor_top3_acc
semantic_anchor_gt_rank
semantic_anchor_coarse_goal_error_m
human_anchor_distance_m
human_anchor_projection_error_m
human_anchor_turn_rate
human_anchor_yaw_rate
human_anchor_landmark_rate
human_anchor_goal_rate
```

Trajectory endpoint metrics now measure the predicted **core anchor**, not the
final destination.

The existing final-goal heatmap diagnostics remain separate:

```
heatmap_top1_acc
heatmap_top3_acc
heatmap_coarse_goal_error_m
gt_frequency / pred_frequency / mean_probability / confusion_counts
```

This separation allows later experiments to tell whether a failure comes from
long-range grounding, core-anchor inference, local trajectory geometry, or
stop/finalization.
