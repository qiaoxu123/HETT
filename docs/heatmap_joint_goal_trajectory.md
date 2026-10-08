# HETT Heatmap + Multi-Trajectory: Learned Joint Goal/Path Selector (Phase 2)

## Why this branch exists

Frozen SBF selector paired with HETT improved **Seen SR 27.34% -> 40.63%** but reduced **Unseen SR 25.39% -> 19.14%**; Seen SPL also did not improve. This is a motivation, **not a result of this branch**.

Start point: `2027-CVPR/hett-heatmap-multitrajectory`. This branch already generates K goal proposals x M trajectory modes with a separate learned Stop Head and GT-conditioned auxiliary teacher imitation. We extend **that existing implementation** rather than porting the frozen SBF selector.

## New behavior

1. HETT 28x28 belief remains the only source of inference goal candidates. Greedy NMS produces K proposals in normalized (x,y) coordinates.
2. For each goal, existing HETT spatial features, relative (goal - pose), and heatmap prior feed a small **residual goal scorer**. Its final layer is zero-initialized, so before training it reproduces prior goal ordering.
3. The joint goal/path distribution is
   `log_softmax(log P_heatmap(goal) + goal_residual)` +
   `log_softmax(mode_logits | goal)`.
4. Predicted-goal ranking **is supervised only when at least one predicted candidate is within success_dist of the GT**; otherwise no false-positive candidate is labeled.
5. A separate predicted-candidate imitation loss uses the teacher suffix as a **label**, selecting the closest covered predicted goal; the GT-conditioned auxiliary imitation from Phase 1 is retained. GT coordinates/teacher future are **never** inference inputs.
6. Independent Stop Head is unchanged by default; `--trajectory_disable_learned_stop` isolates controller/arrival effects. Learned stop still requires predicted endpoint proximity.
7. Existing stage1/stage2 and default heatmap Top-1 execution remain untouched unless `--trajectory_use_for_control` is explicitly set.

**Controls**:
- `--trajectory_selector_mode prior` (default): legacy HETT heatmap + path-mode choice, for paired baseline.
- `--trajectory_selector_mode joint`: learned joint goal/path scoring. Only affects trajectory-based control when `--trajectory_use_for_control` is enabled.
- `--trajectory_goal_k 5` (default; try 8 after ensuring training/eval consistency).
- `--trajectory_ranking_loss_weight 0.2`, `--trajectory_candidate_loss_weight 0.3` (initial hypotheses; not validated optimum).
- `--trajectory_disable_learned_stop`: disable stop-only action for failure decomposition.

## Training data and leakage guards

- Human teacher suffix from CityFlight/MTurk is used only for supervised loss and diagnostic ADE/FDE.
- Oracle nearest candidate, GT target and GT stop correctness are **logging/evaluation only**; the network never receives them during inference.
- Candidate ranking is trained on nearest covered goal (20m default success radius), using *student* goal hypotheses. Off-pool examples are masked and must be reported.
- Zero initialization of goal scorer preserves original ranking *before training*; learned rankings must be benchmarked on unseen data. Selection on Val Unseen cannot be used for weight tuning.
- Candidates are still **not** obstacle-validated. Predicted curves are hypotheses; do not claim collision safety.
- Dataset uses normalized map coordinates with world X=columns and Y=rows. Physical distance = normalized distance x `map_meters`.

## Run on local trusted CityNav GPU environment

Inspect `multiagent/defaultpaths.py` and available checkpoint/data before starting.

CPU regressions (no CityNav required):

```bash
python -m pytest -q tests/test_heatmap_trajectory.py tests/test_multi_landmark_relations.py tests/test_heatmap_relative_geometry.py tests/test_dense_spatial_belief.py
python -m py_compile multiagent/models/heatmap_trajectory.py multiagent/models/ET_haa.py multiagent/agent.py multiagent/main.py multiagent/parser.py
```

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=1 BATCH=2 GRID=7 RUN_NAME=joint_smoke \
  bash multiagent/train_1gpu.sh --benchmark_batches 2 --trajectory_goal_k 5
```

Then use **the same trained checkpoint and eval episodes** for ablations:

| Mode | Goal/path ranking | Controller | Stop | Purpose |
|---|---|---|---|---|
| A | heatmap Top-1 | legacy Stage1/Stage2 | legacy | HETT control baseline |
| B | prior goal + path mode | multitrajectory | learned | trajectory effect |
| C | learned joint | multitrajectory | learned | learned reranking effect |
| D | learned joint | multitrajectory | disabled | stop failure decomposition |

Training for A may use `--no_heatmap_trajectory` as a clean ablation, but it must be a matched-start comparison. B/C/D should use the **same checkpoint** and differ only in inference flags. No test_unseen during tuning.

Example short run (not a performance claim):

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=10 BATCH=8 GRID=7 RUN_NAME=heatmap_joint_10e \
  bash multiagent/train_1gpu.sh --trajectory_goal_k 5
```

For B: `--trajectory_use_for_control --trajectory_selector_mode prior`; C: `--trajectory_use_for_control --trajectory_selector_mode joint`; D: add `--trajectory_disable_learned_stop`. Use a local eval entrypoint and `--checkpoint` to reuse the *same* checkpoint. Do not retrain merely to switch inference selector.

## Measurements

In addition to SR/SPL/OSR/NE and goal TopK coverage:
- `trajectory_prior_hit@20`, `trajectory_joint_hit@20`, `trajectory_oracle_hit@20` (oracle = GT-nearest candidate **coverage diagnostic** only).
- Selected trajectory `prior_FDE_m` / `joint_FDE_m`, minimum ADE/FDE among all proposals.
- `trajectory_rank_supervision_rate`: fraction of evaluated steps with a GT-nearby predicted candidate; essential to assess training signal sparsity.
- `trajectory_stop_precision` and `trajectory_stop_count`: precision among stops actually triggered during evaluated rollout, **not** a substitute for recall/false-stop analysis.
- Evaluate paired per-episode improvements across val_seen / val_unseen and compute bootstrap CIs; record route lengths and action budgets.
- Never compare the GT-using oracle as a deployable method.

## Known issues / next work

- Current goal reranker uses existing HETT spatial features; it does **not** yet have separate landmark-specific relation tokens or observed visual correspondence.
- Candidate imitation uses the demonstrated future route as proxy even if student deviates from teacher state; off-policy recovery requires dedicated labels.
- No dynamic map collision checks or uncertainty calibration are implemented.
- Learned Stop Head remains weakly supervised; OSR-SR gaps and short path execution require closed-loop validation.
- Do not claim SR/SPL gains until actual matched CityNav experiments and hard-negative analysis are completed.
