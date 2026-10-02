# Direct Trajectory Navigation in develop

The learned student policy no longer uses the Stage-1 / Stage-2 controller.
Navigation is driven directly by trajectory hypotheses built on the original
7x7 HETT candidate grid.

## Representation

For each of the 49 HETT candidate regions, the existing heatmap head provides
the trajectory-mode score. A geometry head predicts:

```
endpoint offset:      dx, dy
trajectory residual:  5 x (dx, dy)
```

Each candidate therefore represents:

```
region belief
+ continuous endpoint refinement
+ fixed-horizon trajectory residual
```

The endpoint offset is bounded to half a 7x7 cell in each axis, so each mode
refines continuously inside its coarse region.

## Fixed physical horizons

Each trajectory uses:

```
10 m / 25 m / 50 m / 100 m / final endpoint
```

A straight fixed-horizon anchor is built from the current UAV position toward
the refined endpoint. The residual head is zero-initialized, so early training
starts from stable straight anchors.

## Training

There is one spatial mode-classification objective: the existing 7x7 heatmap
loss.

Trajectory geometry is supervised only on the GT 7x7 mode:

```
L = L_HETT
  + lambda_heatmap * L_heatmap
  + lambda_endpoint * L_endpoint
  + lambda_trajectory * L_trajectory
```

The CityFlight human teacher path is projected continuously from the current
state and sampled at 10/25/50/100 m plus the final goal. Invalid long horizons
near the destination are masked.

The teacher + student rollout schedule is preserved, and BERT/DarkNet remain
trainable. The teacher state generator itself is now aligned with the
trajectory policy: it advances along the recorded CityFlight human trajectory
by a fixed physical arc-length step (default 10 m), preserving interpolated
human x/y/z/yaw instead of using the legacy Stage-1/Stage-2 controller.

For teacher states, trajectory targets use the known human-path arc coordinate
directly rather than re-projecting the pose to the polyline. This avoids
ambiguous supervision at loops or self-intersections. Student states still use
continuous projection onto the human teacher path to obtain recovery targets.

## Teacher rollout

Teacher and student use the same HETT/trajectory network and the same losses.
They differ only in next-state generation:

```
teacher: current human arc -> + teacher_step_m -> interpolated human Pose4D
student: predicted Top-1 trajectory -> first waypoint -> controller move
```

Teacher yaw is no longer randomized. It is interpolated on the shortest wrapped
angular path between recorded human poses, preserving the first-person viewing
behavior in CityFlight.

The teacher rollout is still capped by `max_action_len`; `teacher_step_m`
controls the physical spacing of teacher states.

## Student execution

Student navigation is trajectory-only:

```
7x7 trajectory-mode scores
        ↓
3x3 NMS + Top-K
        ↓
continuous endpoints
        ↓
fixed-horizon trajectories
        ↓
select Top-1 trajectory
        ↓
execute only first ~10 m waypoint
        ↓
observe again and replan
```

There is no learned-policy Stage-1 / Stage-2 switch, recovery threshold, or
fine-navigation branch in student execution.

The legacy Stage-1 / Stage-2 code remains reachable only inside the preserved
teacher rollout path so that the requested teacher training behavior is not
changed.

## Bottleneck diagnostics

Every train epoch and validation split now records structured diagnostics in:

```
navigation_diagnostics.jsonl
```

Each JSONL record also stores raw aggregated 7x7 heatmap data:

```
gt_frequency       # 7x7 GT-cell frequency
pred_frequency     # 7x7 predicted Top-1 frequency
mean_probability   # mean 7x7 probability field
confusion_counts   # 49x49 GT-cell -> predicted-cell confusion matrix
sample_count
```

This makes center bias, mode collapse, spatial blind spots, and systematic
cell-to-cell confusion directly inspectable after training.

Key heatmap statistics:

```
heatmap_top1_acc
heatmap_top3_acc
heatmap_gt_prob
heatmap_top1_conf
heatmap_entropy
heatmap_gt_rank
heatmap_cell_error
heatmap_coarse_goal_error_m
```

Trajectory statistics:

```
trajectory_top1_endpoint_error_m
trajectory_oracle_topk_endpoint_error_m
trajectory_topk_gt_recall
trajectory_first_wp_error_m
```

Progress/stop diagnostics:

```
progress_mae
stop_trigger_rate
premature_stop_rate
near_goal_continue_rate
```

Teacher rollout diagnostics additionally include:

```
teacher_coverage_ratio
teacher_truncated_rate
```

These make it possible to distinguish four common bottlenecks:

1. low heatmap Top-1/Top-3 -> spatial grounding failure;
2. good Top-K but poor Top-1 -> ranking failure;
3. good mode accuracy but large endpoint/first-waypoint error -> geometry failure;
4. good localization but poor SR -> stop/finalization or execution failure.

## Key simplification

Previous:

```
7x7 heatmap -> Stage-1 -> Stage-2
                    or
7x7 -> 28x28 dense belief -> trajectory
```

Current:

```
7x7 HETT candidates
        ↓
score + endpoint offset + trajectory residual
        ↓
Top-K trajectories
        ↓
first waypoint
        ↓
replan
```
