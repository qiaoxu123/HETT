# HETT Joint Goal/Trajectory — Second-pass Failure Audit (2026-10-09)

Review target: `2027-CVPR/heatmap-joint-goal-trajectory-fixes`, descended from the failed epoch-1 `ee9fe962` branch. No new CityNav GPU evaluation has been performed during this review.

## 1. What the original measurements imply

- Full `val_seen` (N=2470): Top-20 Hit@20=90.40%, prior selected goal Hit@20=26.72%, joint selected goal Hit@20=21.73%, SR=3.00%, OSR=3.44%.
- Full `val_unseen` (N=2697): Top-20 Hit@20=89.91%, prior selected goal Hit@20=20.97%, joint selected goal Hit@20=16.42%, SR=2.89%, OSR=3.86%.
- These measurements do **not** establish that the correctly ranked goals actually existed at the later UAV positions. Top-20 coverage is an initial/step-conditioned pool statistic; rank and control are separate failures.
- Official CityNav SR depends on distance of **last recorded pose** to target, NOT on a stop action. OSR counts ever reaching 20m. Therefore `Stop=0` alone cannot explain an OSR around 3–4%.

## 2. High-priority verified defects fixed by the audit branches

| Severity | Confirmed source behavior | Change |
|---|---|---|
| Critical | Flat argmax over goal/mode mixes goal posterior with mode confidence | Marginalize modes to choose goal, then choose its best path mode |
| Critical | At horizon/stagnation, an already-executed terminal displacement could be omitted from scored trajectory | Always append executed terminal movement |
| High | Observer continued recording dead episodes unless they were classified as true stops | Track `ended` explicitly and skip subsequent callback steps |
| High | Original Stop targets used half radius and a teacher-suffix proxy | Use official 20m radius for Stop labels |
| High | Controller follows first path point even when in discrete motor 5m STOP radius | Skip already reached waypoints, use farther valid point |
| High | Student training used legacy control but joint evaluation used trajectory control | Apply configured joint-policy action control during student rollout |
| High | Arrival head considered Top-1 distance instead of selected joint goal | Condition stop on selected-policy goal distance |
| Medium | Rotation-only actions counted as no progress and could trigger trajectory stagnation | Only flag full pose stagnation (no displacement and no turn) |
| Medium | Retry evaluation defaulted to variant C although config names only joint_relation_trajectory | Use configured variant set |
| Medium | `evaluate_every=1` was ignored (only epoch 1 and last were evaluated) | Honor evaluation cadence |
| Medium | Evaluation changed policy flags after epoch and could change training policy for multi-variant runs | Restore the configured training variant |
| Medium | `heatmap_execution=waypoint` silently overrode `trajectory_use_for_control` | Fail fast on conflicting flags |

All fixes above require on-device regression before making any accuracy claim. CPU tests only guard logic, not planning policy quality.

## 3. Unresolved architectural and statistical risks (do not silently treat as fixed)

### P0 — RGB does not directly feed the joint planner

`multiagent/models/ET_haa.py` computes `emb_frames` through Darknet + the legacy ET pathway. But `CompactSpatialBelief` receives `maps`, `emb_lang`, pose/landmark geometry; then `HeatmapTrajectoryHead` receives only belief spatial features, current pose/yaw, instruction and named-landmark geometry/history coordinates. Neither gets `emb_frames` or RGB crop content.

`LandmarkNavMap.to_array()` returns view/exploration coverage and static global/referenced landmark masks — there are no RGB feature maps. Thus the **executed joint trajectory policy is RGB-blind**, although RGB still trains auxiliary legacy direction/progress heads. This is an architecture observation, not evidence that RGB itself is useless.

Suggested *separate* experiment: cheap gated fusion of the already computed ET visual embedding into observed-view spatial features, with a zero-init residual gate; only add candidate-local crops in a later experiment if needed. Benchmark seen/unseen and training speed. Do not claim pixel-to-landmark correspondence unless spatial alignment is measured.

### P0 — Controller bottleneck independently of Stop

Joint selected-goal Hit@20 remains ~16–22% and OSR ~3–4%. Macro actions contain at most `move_iteration=5` discrete actions; a 180-degree turn can consume multiple turn-only micro actions. Eight-waypoint paths are reselected every macro step with no goal hysteresis. Even perfect goal selection may fail with insufficient horizon or motor progress. Inspect `macro_zero_translation_rate`, `mean_goal_progress_m`, `reached_from_outside`, and `initial_already_successful` newly recorded by the metrics layer.

### P1 — Candidate ranking labels are single-positive but official success is radius-based

`candidate_ranking_loss` uses nearest-covered candidate as the only positive and NLL over goal probabilities. Another proposal less than 20m from GT can be a *valid success goal* but is still penalized if not the exact nearest. Compare against multi-positive `-logsumexp` over ALL candidates within 20m, preferably with distance weighting. Add separate loss diagnostic; do not change the existing checkpoint loss and claim apples-to-apples convergence.

### P1 — Predicted-goal path imitation has off-policy/endpoint contradictions

`resample_teacher_suffix` selects human teacher trajectory points nearest the student's *current pose*, appends the GT endpoint and resamples; this is a proxy and may request unrealistic rejoining or travel. Separately, a predicted goal within 20m gets a teacher path ending at **GT**, while the generated predicted-goal path is hard-clamped to the **candidate endpoint**. That creates unavoidable endpoint mismatch and may teach the mode head to score the wrong paths. Inspect teacher deviation and supervise only the shared valid prefix or make labels endpoint-consistent in a distinct ablation.

### P1 — Stop is blocked in two stages

The positive class was previously unnecessarily strict. Even after using the correct official radius, effective training weight `0.05`, `pos_weight=2`, and inference threshold `0.8` may be poorly calibrated. Actual stop also requires selected predicted endpoint within `success_dist`. A wrong selected endpoint suppresses Stop even when UAV is truly within GT radius. Audit logs now include `trajectory_stop_positive_count`, `trajectory_stop_supervised_count`, and `trajectory_arrival_gate_blocked` (GT used only for diagnosis). Calibrate on train/val_seen; validate on val_unseen without selecting weights there.

### P1 — New ranking gradients can disturb the high-recall heatmap

`log_goal` is gathered from differentiable heatmap probabilities; joint ranking NLL can backpropagate into belief/backbone parameters, even though the top-K index selection is nondifferentiable. Without phased warming of new heads/frozen belief, training may destroy the ~90% Top-20 candidate coverage. Test frozen-belief head warmup vs end-to-end, monitor initial Top-20 coverage each epoch, and check actual gradient norms grouped by head/base.

### P2 — Named-landmark semantic grounding is under-diagnosed

The map supplies annotated contours and matched reference names. `build_landmark_batch` falls back to whole-instruction context for unmatched mention spans. A selector can then infer mostly location/map priors instead of correctly applying direction words, attribute constraints, or image evidence. Log matched/total/truncated name counts by split and evaluate name-to-landmark permutation and direction-flip controls.

### P2 — Long-range visual memory is only partial

Legacy ET stores aggregated per-grid visual embeddings, but the new candidate relation selector only gets up to 10 past normalized (x,y) positions, plus current heading; it does not consume ET's stored visual feature history, human final arrival heading, nor optical correspondence. Explicitly distinguish *pose history* from *visual history* in paper claims.

## 4. Minimal matched diagnostic matrix before another expensive multi-epoch run

Keep checkpoint, episode set, split, seed and action horizon fixed for all test-time ablations:

| Case | Goal choice | Controller | Stop | Question |
|---|---|---|---|---|
| A | Heatmap prior | Legacy two-stage | Legacy | What is preserved HETT performance? |
| B | Heatmap prior | Fixed multi-trajectory | Off | How much is lost in the trajectory controller? |
| C — diagnostic upper bound only | GT-nearest within the **same predicted Top-20** | Fixed multi-trajectory | Off | Can this controller reach the right candidate if selection is solved? |
| D | Joint relation goal | Fixed multi-trajectory | Off | Does joint ranking improve target arrival? |
| E | Joint relation goal | Fixed multi-trajectory | Learned | What additional effect does Stop have? |

Case C MUST remain an oracle-labelled offline diagnostic, never be used as an inference/deployable policy, and never be included as a claimed model result. Do not compare a previous 256-episode Seen number to the full 2470-episode result as a controlled causal improvement.

- If C still has low OSR, prioritize trajectory/motor/horizon rather than larger semantic networks.
- If C improves but B/D remain low, prioritize candidate-level semantics and goal ranking.
- If D has high OSR and E has low final SR, then focus on arrival/stop and terminal drift.
- Report `initial_already_successful`, `success_from_outside`, `reached_from_outside`, `initial_heatmap_top20_hit20`, and `macro_zero_translation_rate`. If most SR comes from starting within 20m, the agent has not learned actual approach navigation.

## 5. Reproducibility and status

Use new output folders and the original common initialization checkpoint. Re-evaluating old trained weights under *controller/metrics-only* repairs can isolate bugs, but newly changed Stop targets / on-policy training / architecture need fresh training. New fixes have CPU GitHub Actions checks; **NO new GPU smoke or full CityNav evaluation performed here**. Avoid continuing the existing SR~3% epoch-1 checkpoint as though it validated the new learning objective.
