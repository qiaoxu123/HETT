# HETT Navigation Failure Diagnosis

**Source under test:** `2027-CVPR/hett-sequential-backward` at `501f25af9991dd8f2d30fb588622f77357de1589`  
**Diagnostic branch:** `diagnosis/navigation-failure-501f25af`  
**Date:** 2026-10-09 (Asia/Shanghai)

## Executive finding

The strongest measured failure signal is a **selector gap**: on the fixed random validation sample, the Top-20 candidate pool contained a goal within the 20 m success radius at 92.4% of active Seen steps and 83.0% of active Unseen steps for the current joint trajectory controller. The selected joint goal was within 20 m on only 27.8% and 16.7% of those steps, respectively. The GT-only Top-20 oracle raised success to 96.9% Seen and 95.3% Unseen. This is an upper bound; the oracle is isolated to C and does not change any other rollout.

Execution is a second bottleneck. On these same 128-per-split episodes, Heatmap Prior plus bounded 50 m waypoint execution (A) reached 28.9%/18.0% SR, while the current joint trajectory controller with Stop off (D) reached 14.1%/15.6%. A vs D is +14.8 percentage points Seen and +2.3 Unseen. Joint ranking did not improve over pure Heatmap Prior in Seen (A=B=28.9%) and was +1.6 points in Unseen (B=19.5% vs A=18.0%), too small to establish a robust selector advantage on this sample.

Stop was not an independent failure cause in this checkpoint: E and D were identical in all scored metrics on both splits; there were zero policy stop decisions. The learned stop head never reached the action threshold plus endpoint-distance gate.

The reported 3–4% OSR is **not reproduced** by the present common checkpoint and fixed sample: current joint D OSR is 16.4% Seen and 20.3% Unseen. Prior full-validation work from an earlier trajectory checkpoint/code version did show OSR collapsing to 5.9–7.7% under learned-path control versus 39.9–52.0% for the base heatmap route. Therefore, treat the user's 3–4% as version/checkpoint-specific; it is consistent with execution/selection collapse, but the exact value needs the original 20-epoch checkpoint evaluated under this branch.

## Source audit

### Rollout, selectors, controller, and Stop

- `multiagent/agent.py` computes the heatmap Top-K, trajectory proposals, selector indices, actual executed waypoint, and Stop decisions. During student evaluation, the GT trajectory and goal are only consumed by diagnostics/metrics; only C explicitly reads the GT goal to select the nearest proposal.
- `multiagent/models/heatmap_trajectory.py` generates endpoint-conditioned paths and forcibly replaces the final waypoint with the candidate goal. Consequently, path FDE has a hard lower bound equal to candidate-goal endpoint error; residual waypoint prediction cannot remove that terminal bias.
- The existing `prior` selector is not pure Heatmap Prior: it combines candidate heatmap mass and mode logits. The diagnosis adds an explicit pure-prior branch for A/F (Top-1 heatmap endpoint, best mode) and resets diagnostic flags between variants.
- `multiagent/models/candidate_relation_selector.py` encodes language, landmark names, and relative candidate geometry. The shared trained checkpoint predates that selector, so relation selection is disabled in all six variants; its randomly initialized weights are never evaluated.
- `multiagent/teacher/algorithm/lookahead.py` computes desired heading with `atan2(dy, dx)`, wraps relative yaw, and emits forward/turn actions. Waypoint movement uses the same x/east and y/north convention as map goals.
- `multiagent/env.py` maps world x to normalized x and flips world y into raster row coordinates; `unnormalize_position` is the inverse. Candidate relation geometry uses east=`dx`, north=`-dy`. The inspected coordinate transforms are mutually consistent; no x/y or scale-sign inversion was found.
- The current learned controller picks a local predicted waypoint and runs the discrete lookahead mover. A bounded waypoint diagnostic calls `bounded_heatmap_step` directly with a 50 m cap. Both get the same 20 macro-step horizon. No SBFNav weights are loaded.
- Stop requires both probability above threshold and predicted endpoint within the success radius. Stop false negatives can remain because of this endpoint gate; in the current sample the policy never fired at all.

### Definite bug fixed

Trajectory stagnation used `cos(new_yaw - old_pose.y)` where `old_pose.y` is a position in meters. It must use `old_pose.yaw` in radians. This could corrupt no-progress detection. The code now calls `wrapped_yaw_delta_rad(new_yaw, old_yaw)`, with tests for zero yaw difference and wraparound at ±π. This bug changes stagnation accounting, but does not explain the large candidate-selection gap.

### BERT, learning rate, and Teacher heading

- `CustomBERTModel` loads `bert-base-uncased`; freeze calls are commented out. The audited language encoder has 109,534,641 parameters and is trainable. The saved training configuration used Adam at **1e-4** with weight decay 0; the language, vision, and navigation optimizers all receive this same learning rate. That is high for full BERT fine-tuning compared with common encoder fine-tune settings, but the present experiments do not establish that LR as the primary SR cause.
- `env._get_obs(random_direction=True)` randomizes yaw with probability 0.5 each teacher observation. Teacher rollout calls this at reset and after every step. The heading label is recomputed from the randomized pose, so the geometry remains self-consistent per step, but the temporal observation heading distribution differs from student evaluation and can add motion discontinuity.

## Closed-loop experiment

All A–F use the same checkpoint, same seed (0), same per-split 128 randomly sampled episode IDs, same 20-step maximum, 20 m success radius, and K=20 evaluation proposals. The same episodes are reused for all variants. C is the only GT-selected arm.

**Checkpoint:** `hett-heatmap-joint-goal-trajectory/artifacts/joint_goal_trajectory/run1/checkpoints/epoch01.pt`  
**SHA-256:** `06d5e26474dea2bed92f995c4a46d8a0e87416282c8753d37421b05324d21758`  
**Checkpoint note:** this is an existing one-epoch trajectory-trained checkpoint with trained trajectory and joint-scoring heads. The repository's older 20-epoch base checkpoint does not contain the added trajectory head, so using it for D/E/F would leave those heads randomly initialized. The relation selector added after the shared checkpoint is disabled. The checkpoint trained with K=5 while the ablation evaluates K=20, so K>5 selector scores are an evaluation distribution shift.

| Arm | Selector | Controller | Stop |
|---|---|---|---|
| A | Pure Heatmap Prior | Bounded waypoint, max 50 m | Off |
| B | Joint | Bounded waypoint, max 50 m | Off |
| C | Top-20 nearest-to-GT oracle | Bounded waypoint, max 50 m | Off |
| D | Joint | Current trajectory controller | Off |
| E | Joint | Current trajectory controller | On |
| F | Pure Heatmap Prior | Current trajectory controller | Off |

### Fixed random 128 per split: primary comparison

Percentages are shown as percentages; NE and distances are meters. nDTW/sDTW use Euclidean DTW over recorded world-coordinate paths and the 20 m success radius. Candidate `Pool@20m` is the active-step Top-20 pool hit rate; `Selected@20m` is the corresponding selector hit rate.

| Split | Arm | SR | SPL | OSR | NE | nDTW | sDTW | Heatmap Top-5 Hit@20 | Pool@20m | Selected@20m | Mean steps to first enter | Enter then leave | Stop P/R | Timeout / stagnation | sec/episode | peak VRAM allocated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen | A | 28.91 | 28.40 | 32.81 | 43.79 | 0.240 | 0.090 | 64.6 | 92.7 | 28.9 | 3.83 | 3.9% | N/A / 0% | 128 / 0 | 0.437 | 0.79 GB |
| Seen | B | 28.91 | 28.40 | 33.59 | 42.04 | 0.244 | 0.090 | 66.2 | 91.5 | 29.0 | 3.79 | 4.7% | N/A / 0% | 128 / 0 | 0.430 | 0.79 GB |
| Seen | C | 96.88 | 91.44 | 97.66 | 11.57 | 0.285 | 0.279 | 90.3 | 96.1 | 96.1 | 3.92 | 0.8% | N/A / 0% | 128 / 0 | 0.427 | 0.79 GB |
| Seen | D | 14.06 | 13.77 | 16.41 | 51.36 | 0.298 | 0.058 | 65.5 | 92.4 | 26.3 | 14.10 | 2.3% | N/A / 0% | 85 / 43 | 0.393 | 0.79 GB |
| Seen | E | 14.06 | 13.77 | 16.41 | 51.36 | 0.298 | 0.058 | 65.5 | 92.4 | 26.3 | 14.10 | 2.3% | N/A / 0% | 85 / 43 | 0.398 | 0.79 GB |
| Seen | F | 16.41 | 16.24 | 17.19 | 52.02 | 0.292 | 0.073 | 65.2 | 92.2 | 26.7 | 14.55 | 0.8% | N/A / 0% | 80 / 48 | 0.391 | 0.79 GB |
| Unseen | A | 17.97 | 17.62 | 23.44 | 54.53 | 0.228 | 0.059 | 52.6 | 82.0 | 17.6 | 3.97 | 5.5% | N/A / 0% | 128 / 0 | 0.426 | 0.79 GB |
| Unseen | B | 19.53 | 19.09 | 25.78 | 51.97 | 0.233 | 0.063 | 52.4 | 82.7 | 18.9 | 4.00 | 6.2% | N/A / 0% | 128 / 0 | 0.433 | 0.79 GB |
| Unseen | C | 95.31 | 89.60 | 96.88 | 10.93 | 0.301 | 0.286 | 84.7 | 92.7 | 92.7 | 4.22 | 1.6% | N/A / 0% | 128 / 0 | 0.438 | 0.79 GB |
| Unseen | D | 15.63 | 14.97 | 20.31 | 60.98 | 0.298 | 0.084 | 52.4 | 83.0 | 17.2 | 11.50 | 4.7% | N/A / 0% | 90 / 38 | 0.395 | 0.79 GB |
| Unseen | E | 15.63 | 14.97 | 20.31 | 60.98 | 0.298 | 0.084 | 52.4 | 83.0 | 17.2 | 11.50 | 4.7% | N/A / 0% | 90 / 38 | 0.408 | 0.79 GB |
| Unseen | F | 14.06 | 13.47 | 16.41 | 62.28 | 0.291 | 0.068 | 52.7 | 82.8 | 15.9 | 11.29 | 2.3% | N/A / 0% | 81 / 47 | 0.401 | 0.79 GB |

`Selected@20m` for A/B/C/D/E/F is the actual selected candidate's GT hit fraction over active steps; C uses GT only to pick the nearest available proposal. The first-enter statistic includes starting inside the radius; only 0 Seen and 3 Unseen sampled episodes started within 20 m for A.

The `Top-5 Hit@20m` and candidate hits are measured over each policy's active states, not a single shared frozen state distribution. Initial-state hits should be taken from the per-episode JSON when doing a strict selector-only comparison. See the full per-episode traces and `summary.csv` under `artifacts/navigation_failure_diagnosis/closed_loop_128_random/`.

Operational movement statistics (distance columns in meters; turns count macro heading changes above 15 degrees; zero movement is based on consecutive observed poses):

| Split | Arm | Initial dist | Final dist | Min dist | Net approach | Path length | Mean/min flight step | Turn count | Zero movement | Goal switches | Foldback rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen | A | 190.8 | 43.8 | 39.8 | 147.0 | 174.1 | 9.16 / 0.0 | 1.08 | 78.9% | 102 | 1.4% |
| Seen | B | 190.8 | 42.0 | 37.1 | 148.8 | 178.6 | 9.40 / 0.0 | 1.16 | 78.4% | 107 | 1.7% |
| Seen | C | 190.8 | 11.6 | 11.0 | 179.3 | 200.1 | 10.52 / 0.0 | 1.63 | 76.0% | 276 | 5.4% |
| Seen | D/E | 190.8 | 51.4 | 48.5 | 139.5 | 159.0 | 8.38 / 0.0 | 3.83 | 14.0% | 385 | 0.0% |
| Seen | F | 190.8 | 52.0 | 50.0 | 138.8 | 156.0 | 8.33 / 0.0 | 3.63 | 15.4% | 334 | 0.0% |
| Unseen | A | 205.7 | 54.5 | 47.3 | 151.2 | 188.9 | 9.94 / 0.0 | 1.22 | 77.2% | 98 | 0.4% |
| Unseen | B | 205.7 | 52.0 | 44.0 | 153.8 | 193.8 | 10.20 / 0.0 | 1.27 | 76.7% | 110 | 0.8% |
| Unseen | C | 205.7 | 10.9 | 10.1 | 194.8 | 213.9 | 11.25 / 0.0 | 1.65 | 74.8% | 268 | 5.4% |
| Unseen | D/E | 205.7 | 61.0 | 55.6 | 144.8 | 171.8 | 9.05 / 0.0 | 4.10 | 12.3% | 356 | 0.0% |
| Unseen | F | 205.7 | 62.3 | 58.1 | 143.5 | 167.0 | 9.03 / 0.0 | 3.98 | 13.9% | 302 | 0.0% |

The displayed minimum flight step is zero for each arm because some candidate waypoints coincide with the current pose. `per_episode.csv` includes per-episode min/mean flight step, zero-displacement share, switches, and foldback; `steps` in each JSON contains every macro displacement and yaw delta.

### Full validation follow-up

The fixed sample showed a large Seen A-vs-D gap, so full validation A and D were evaluated with `fast_eval` on the same model/checkpoint, all 2,470 Seen and 2,697 Unseen episodes, and batch size 8. This follow-up keeps Top-20 candidate ordering and the SBF-style bounded waypoint policy fixed against the current joint selector and trajectory controller, respectively. A's full-set SR and OSR are materially higher on both splits:

| Split | Arm | SR | SPL | OSR | NE (m) | sec/episode | peak allocated GPU |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A | 25.83 | 25.22 | 29.07 | 44.07 | 0.0771 | 1.18 GB |
| Seen | D | 13.04 | 12.81 | 14.94 | 51.64 | 0.0802 | 1.18 GB |
| Unseen | A | 20.62 | 20.11 | 24.84 | 48.92 | 0.0762 | 1.18 GB |
| Unseen | D | 11.46 | 11.18 | 14.50 | 56.10 | 0.0807 | 1.18 GB |

Paired episode bootstrap (10,000 resamples, seed 50125) for A minus D SR: Seen **+12.79 pp** (95% percentile CI 11.09 to 14.62; 416 episodes gained, 100 lost); Unseen **+9.16 pp** (CI 7.56 to 10.75; 382 gained, 135 lost). This confirms a substantial current-controller penalty across the full validation sets. It does not isolate trajectory-model quality from the waypoint policy's different execution behavior. Full per-episode outputs are in `artifacts/navigation_failure_diagnosis/full_A_D/`.

### Existing full-validation evidence from the earlier one-epoch trajectory run

An earlier full-validation result on a pre-relation-selector checkpoint had different training K and controller code. It is useful corroboration, not a direct estimate for the checkpoint above:

| Split | Base heatmap route A SR / OSR | Prior learned path B SR / OSR | Joint learned path C SR / OSR | Joint Stop-off D SR / OSR |
|---|---:|---:|---:|---:|
| Seen (2,470) | 29.47 / 52.02 | 4.90 / 5.91 | 5.79 / 6.92 | 5.79 / 6.92 |
| Unseen (2,697) | 18.54 / 39.93 | 5.67 / 7.30 | 5.90 / 7.75 | 5.90 / 7.75 |

That run showed Top-20 coverage stayed materially higher than learned-selected path execution, and Stop-off equaled Stop-on. Its A policy was not the current A/B bounded-waypoint policy, so do not combine these rows statistically with the fixed-sample table.

## Training supervision and optimization

### Ranking coverage and target bias (measured)

A separate no-update teacher-forced audit used 128 seed-0 `train_seen` episodes, K=5 (the training setting), 20 steps, and the same checkpoint. No backward pass, optimizer step, or weight change occurred.

- Teacher-supervised active states: 2,560.
- Ranking-valid states with at least one proposal within 20 m: 2,134 / 2,560 = **83.36%**. The remaining **16.64%** have no positive candidate and produce no ranking-loss gradient.
- Mean nearest candidate endpoint distance to GT: **18.09 m**.
- Proposal minADE: **13.82 m**; minFDE: **18.09 m**. MinFDE equals nearest candidate-goal error because every candidate path's final waypoint is hard-set to its candidate goal.
- Prior/joint selected candidate hit rate over teacher states: **63.91% / 63.83%**; oracle pool hit: **83.36%**.
- Stop-positive teacher labels: **59.92%** of active states; evaluated Stop still made zero decisions.

The current `candidate_ranking_loss` is single-positive NLL: it finds the nearest goal candidate and supervises it only if within 20 m. A distance-soft-label listwise objective would distribute probability across multiple candidates (for example, with a temperature-controlled `softmax(-distance/tau)`) and preserve graded distance information. This is a different objective; it has not been trained or evaluated here.

### Synchronized per-episode inference profile

Current-controller D was profiled on the same fixed 128 IDs per split with batch size 1 and explicit device synchronization around each episode. These timings include the navigation policy/model calls across an episode (not just one forward pass); peak GPU allocation was 0.787 GB and reservation 2.77 GB.

| Split | min (s) | mean (s) | variance (s²) | max (s) |
|---|---:|---:|---:|---:|
| Seen | 0.0975 | 0.3604 | 0.0399 | 2.4923 |
| Unseen | 0.0948 | 0.3372 | 0.0038 | 0.3959 |

Five slowest profiled episodes (all terminated at the action horizon): `cambridge_block_28|270|4` (2.4923 s), `cambridge_block_20|189|3` (0.5135 s), `cambridge_block_20|5|5` (0.4543 s), `birmingham_block_1|566|5` (0.4387 s), `cambridge_block_19|543|0` (0.4243 s). The first episode is a clear timing outlier and likely includes one-time runtime initialization; raw values are retained. Per-episode timings are in `artifacts/navigation_failure_diagnosis/profile_D_128/per_episode.csv`.

### Gradient, BERT, memory, and speed audit

Available checkpoint training log from its one-epoch run: 21,877 episodes, 2,735 batches, 7,175.8 s, peak allocated/reserved GPU memory 63.24/64.12 GB at training batch size 8. First two recorded trajectory-head gradient norms were:

| Group | Batch 1 | Batch 2 |
|---|---:|---:|
| ET base (aggregate) | 26.43 | 23.56 |
| Trajectory context | 4.3e-8 | 2.03e-4 |
| Mode embedding | 0.0837 | 0.0846 |
| Residual waypoint head | 0.0340 | 0.0422 |
| Mode score | 1.677 | 1.752 |
| Goal scorer | 0.0606 | 0.0452 |
| Stop head | 0.0318 | 0.1048 |

This checkpoint's saved audit did not split ET base gradients into Darknet, heatmap, and selector groups, and did not record BERT gradients; those per-module gradient norms cannot be reconstructed from the saved scalar logs. The synchronized profile above records per-episode latency and slowest sample IDs.

No BERT-freeze experiment was trained. The source leaves all 109.5M BERT parameters trainable at LR=1e-4. Freezing may reduce optimizer/backward cost, but it cannot be judged for navigation quality without the requested matched one-epoch training ablation; no quality claim is made from a frozen-at-inference switch.

## Ten representative current-controller failures

The compact case file contains 10 D/E failures, with episode ID, start and target coordinates, selected goal ID sequence, path coordinates, Top-20 hit indicators, first-entry/min/final distance, and termination reason:

- [failure_cases_top10.json](../artifacts/navigation_failure_diagnosis/closed_loop_128_random/failure_cases_top10.json)
- Full per-episode trajectories and candidate/stop metrics: [A–F results](../artifacts/navigation_failure_diagnosis/closed_loop_128_random/)

## Answers and priorities

1. **Why can OSR collapse toward 3–4%?** The measured failure signature is that the Top-20 heatmap pool often contains a valid endpoint while the learned selector picks a distant candidate, then the trajectory controller takes repeated local actions toward endpoints with 42–54 m selected-plan FDE. An earlier full run measured OSR 5.9–7.7%, versus 39.9–52.0% for its base route. The exact 3–4% was not reproduced with this checkpoint/sample.
2. **Is the trajectory controller weaker than bounded waypoint execution?** Yes. On the 128-episode sample, A−D was +14.8 pp SR and +16.4 pp OSR Seen, and +2.3 pp SR and +3.1 pp OSR Unseen. Full validation confirms it: +12.79 pp Seen SR (paired 95% CI 11.09–14.62) and +9.16 pp Unseen SR (CI 7.56–10.75), with OSR +14.13 pp and +10.34 pp, respectively.
3. **Is Joint Selector worse than Heatmap Prior?** Not consistently. It ties A Seen and is +1.6 pp Unseen under bounded control. With the current controller F is +2.3 pp Seen but −1.6 pp Unseen against D. Selected-hit rates stay close; no robust Joint advantage is established. A key problem is both learned-selection arms have a large gap from Top-20 oracle coverage.
4. **Is Stop an independent bottleneck?** No measured effect: D=E and zero policy stops. Investigate head calibration/endpoint gate after selector and path execution.
5. **Is freezing BERT reasonable?** It is a plausible compute ablation because 109.5M parameters are currently trainable at 1e-4, but quality and actual cost savings remain unmeasured. Run the matched one-epoch freeze study before deciding.
6. **Three modules to prioritize:** (1) candidate/goal selection supervision and calibration; (2) trajectory control/execution, including waypoint step and goal switching; (3) teacher-to-candidate supervision coverage and endpoint loss design. Stop is fourth; BERT freeze is a training-cost experiment rather than the first SR fix.

## Reproduction and artifacts

```bash
cd multiagent
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python \
  ../scripts/evaluate_joint_goal_trajectory_checkpoint.py \
  --config ../configs/diagnosis/closed_loop_128.json \
  --checkpoint /home/rental/20260922_1/Workspace/hett-heatmap-joint-goal-trajectory/artifacts/joint_goal_trajectory/run1/checkpoints/epoch01.pt \
  --output ../artifacts/navigation_failure_diagnosis/closed_loop_128_random --limit 128
```

Finalization command: `python scripts/finalize_navigation_failure_diagnosis.py artifacts/navigation_failure_diagnosis/closed_loop_128_random`. Episode IDs and checkpoint hash are stored in `retry_manifest.json` and `COMPLETE.json`. The original `2027-CVPR/hett-sequential-backward` worktree and its preexisting uncommitted files were not modified.
