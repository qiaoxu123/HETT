# HETT Compact Goal–Path Soft Ranking (experimental)

Base: `2027-CVPR/hett-sequential-backward` @ `501f25af9991dd8f2d30fb588622f77357de1589`.
Branch: `2027-CVPR/hett-compact-goal-path-soft`.

## Design: two learning objectives; keep a fixed waypoint controller

**Inputs:** existing BERT language token encoding, landmark geometry, RGB/map features, current pose and causal trajectory history. Frozen pretrained BERT/visual/ET backbone for this selector-only experimental config. The existing HETT Spatial Belief supplies a 28×28 heatmap, NMS selects Top-20 goals.

**One shared candidate selector** (reuse existing `trajectory_head.context`, `goal_scorer`, `candidate_relation`, `mode_score`; checkpoint keys remain compatible):

- **Goal:** rank the 20 heatmap candidates using the current heatmap prior plus learned semantic/geometric evidence. Train soft cross-entropy over target distances to all candidates, temperature 20 m. Ranking gradients do **not** enter the spatial belief/heatmap. Only train-time labels use GT.
- **Path:** for each candidate, form three fixed local anchors (left/straight/right) with a 20 m metric horizon; no learned trajectory residual and no learned terminal heading. For the *nearest GT-covered candidate* only, train a soft cross-entropy over mode logits using (0.7 × human-local-waypoint error + 0.3 × remaining goal distance), temperature 5 m. All distances are measured in meters. Path supervision is ignored when the candidate pool has no goal inside the 20 m success radius; goal-soft supervision still applies.
- Goal-first choice: select goal by `logsumexp(joint_logits, modes)`, then select a mode conditional on that goal. Never flatten goal×mode and use mode peak to choose the global goal.
- **Fixed waypoint control:** drive to the selected **local anchor endpoint** using `bounded_heatmap_step` (no direct rollout of a learned 8-point trajectory). Replan after every observation. Learned stop, learned path residual and human terminal-yaw supervision are disabled in compact mode. The legacy controller/trajectory/head remain available when compact mode is false.

**Important limitation:** waypoint controller bounds XY displacement but is not a full collision-aware dynamic UAV simulator; no obstacle feasibility is asserted. The sampled human suffix supplies a *local training label* and is not guaranteed to be a valid recovery route when a student has strayed from a demonstrated path. This first-stage version does not fix this automatically. Path shape is a fixed candidate bank; the controller executes its local endpoint rather than tracing all intermediate samples.

## Trainable parameters and performance protection

`trajectory_compact_mode=true` freezes BERT, Darknet and **all ET parameters except the current `trajectory_head`**. Head submodules not used in compact mode (learned stop and residual) receive no gradients. Heatmap candidate features and prior are explicitly detached before candidate scoring. Existing optimizers are retained for checkpoint compatibility; frozen groups have no gradient updates. Sequential teacher/student backward remains enabled.

**Losses:**
```text
L = goal_soft * 1.0 + path_soft * 0.5
```
Legacy directional/progress/goal regression, full teacher-conditioned trajectory WTA, learned stop and candidate imitation loss are not included when compact mode is on. No GT target or human suffix enters model inference.

### Controlled experiment

`configs/experiments/hett_compact_goal_path_1e.json` defines exactly **three inference variants** from the same trained checkpoint:

| Variant | Goal choice | Controller |
|---|---|---|
| A_heatmap_waypoint | HETT Heatmap Top-1 | Original bounded heatmap waypoint |
| B_goal_soft | Shared goal-soft selector | Original bounded global-goal waypoint |
| C_goal_path_soft | Same goal-soft selector | Selected 20 m local anchor via bounded waypoint |

For a training ablation that isolates the path-loss contribution, **train a second checkpoint** with `trajectory_candidate_loss_weight:0.0` using exactly the same initial checkpoint, seed and training budget. Switching B and C inference flags on a single checkpoint is *not* a path-loss training ablation. Evaluate on both full `val_seen` and `val_unseen`, compare SR/SPL/OSR/NE against A and prior SBFNav checkpoint results. Do not claim a gain from 8-episode smoke.

Known prior result: under one earlier joint-trained checkpoint, traditional Top-1 execution achieved 29.47%/18.54% SR on seen/unseen full splits, while prior + learned trajectory controller fell to 4.90%/5.67%. This is **previous checkpoint evidence**, not a result of this new branch.

## Smoke command

```sh
git fetch origin
git switch 2027-CVPR/hett-compact-goal-path-soft
git pull
export HETT_INITIAL_CHECKPOINT=/absolute/path/to/initial.pt
export HETT_BASE_ARGUMENTS=/absolute/path/to/base_arguments.json

CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_goal_trajectory_experiment.py \
 --config configs/experiments/hett_compact_goal_path_1e.json \
 --output artifacts/compact_goal_path_smoke --smoke
```

The smoke runs 2 optimizer steps at batch 2 and 8, then an 8-episode diagnostic validation. Check finite loss, gradient and head update audit, memory, no GT inputs to inference, both splits. Next, run **one epoch**, using a fresh output directory and remove `--smoke`.

The existing experiment runner uses a 30% seen-SR first-epoch gate and can stop after epoch 1. This file explicitly sets `epochs=1`; do not interpret the gate as a proven performance target or auto-resume long training. The project is an *experiment* until GPU smoke and full-split validation are reported.
