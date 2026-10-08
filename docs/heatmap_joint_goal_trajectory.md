# HETT Heatmap + Multi-Trajectory: Learned Joint Goal/Path Selector (Phase 2)

## Why this branch exists

Frozen SBF selector paired with HETT improved **Seen SR 27.34% -> 40.63%** but reduced **Unseen SR 25.39% -> 19.14%**; Seen SPL also did not improve. This is a motivation, **not a result of this branch**.

Start point: `2027-CVPR/hett-heatmap-multitrajectory`. This branch already generates K goal proposals x M trajectory modes with a separate learned Stop Head and GT-conditioned auxiliary teacher imitation. We extend **that existing implementation** rather than porting the frozen SBF selector.

## New behavior

1. HETT 28x28 belief remains the only source of inference goal candidates. Greedy NMS produces K proposals in normalized (x,y) coordinates.
2. For each candidate, existing HETT spatial features, current UAV relative pose and heatmap prior feed a small residual goal scorer.
3. **SBF-inspired candidate-level relation reasoning** (not weight or network transfer): a *separate* `CandidateRelationSelector` scores each Top-K candidate against its own geometry to **individually named** referenced landmarks (extent, range, bearing, inside-footprint flag), BERT-contextual name spans and the full instruction (candidate-to-text cross-attention). It also uses **causally observed UAV pose history** (previously visited locations and progress) as a candidate-level descriptor. Landmark attention with mean/soft-min discourages satisfaction of only one of multiple references. If no valid referenced landmark exists, the relation evidence is exactly zero.
4. The joint goal/path distribution is
   `log_softmax(log P_heatmap(goal) + goal_residual + tanh(relation_gate) * relation_evidence)` +
   `log_softmax(mode_logits | goal)`.
   The goal scorer and path residual initialize to preserve their HETT prior behavior. The relation gate starts at `tanh(gate)=0.1` so relation-encoder parameters receive gradients on the first update.
4. Predicted-goal ranking **is supervised only when at least one predicted candidate is within success_dist of the GT**; otherwise no false-positive candidate is labeled.
5. A separate predicted-candidate imitation loss uses the teacher suffix as a **label**, selecting the closest covered predicted goal; the GT-conditioned auxiliary imitation from Phase 1 is retained. GT coordinates/teacher future are **never** inference inputs.
6. Independent Stop Head is unchanged by default; `--trajectory_disable_learned_stop` isolates controller/arrival effects. Learned stop still requires predicted endpoint proximity.
8. Existing stage1/stage2 and default heatmap Top-1 execution remain untouched unless `--trajectory_use_for_control` is explicitly set.

**Controls**:
- `--trajectory_selector_mode prior` (default): legacy HETT heatmap + path-mode choice, for paired baseline.
- `--trajectory_selector_mode joint`: learned joint goal/path scoring. Only affects trajectory-based control when `--trajectory_use_for_control` is enabled.
- `--trajectory_relation_selector` (default enabled) / `--no_trajectory_relation_selector`: include/ablate candidate-specific language/landmark/history evidence **without changing the heatmap or candidate pool**.
- `--trajectory_goal_k 20` (default for the current candidate-ranking run).
- `--trajectory_ranking_loss_weight 0.2`, `--trajectory_candidate_loss_weight 0.3` (initial hypotheses; not validated optimum).
- `--trajectory_disable_learned_stop`: disable stop-only action for failure decomposition.

## Training data and leakage guards

- Human teacher suffix from CityFlight/MTurk is used only for supervised loss and diagnostic ADE/FDE.
- Oracle nearest candidate, GT target and GT stop correctness are **logging/evaluation only**; the network never receives them during inference.
- The relation scorer binds each referenced landmark name to its own contour centroid/extent using instruction token masks, not an aggregate centroid. The short pose history is recorded online from observations only; the last pose is excluded from visited-history distance.
- The learned relation score is **not** the official SBF selector or a faithful SBF reimplementation; only its candidate-specific multi-evidence reasoning principle is adopted. Treat seen/unseen generalization as an empirical question.
- Candidate ranking is trained on nearest covered goal (20m default success radius), using *student* goal hypotheses. Off-pool examples are masked and must be reported.
- Zero initialization of the final goal-scorer layer preserves original ranking before training. The small nonzero relation gate lets its encoder learn immediately; learned rankings must be benchmarked on unseen data. Selection on Val Unseen cannot be used for weight tuning.
- Candidates are still **not** obstacle-validated. Predicted curves are hypotheses; do not claim collision safety.
- Dataset uses normalized map coordinates with world X=columns and Y=rows. Physical distance = normalized distance x `map_meters`.

## Run on local trusted CityNav GPU environment

Inspect `multiagent/defaultpaths.py` and available checkpoint/data before starting.

CPU regressions (no CityNav required):

```bash
python -m pytest -q tests/test_candidate_relation_selector.py tests/test_heatmap_trajectory.py tests/test_multi_landmark_relations.py tests/test_heatmap_relative_geometry.py tests/test_dense_spatial_belief.py
python -m py_compile multiagent/models/candidate_relation_selector.py multiagent/models/heatmap_trajectory.py multiagent/models/ET_haa.py multiagent/agent.py multiagent/main.py multiagent/parser.py
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
| C | learned joint, `--no_trajectory_relation_selector` | multitrajectory | learned | MLP ranking without SBF-inspired relation evidence |
| D | learned joint with candidate relation evidence | multitrajectory | learned | test candidate-specific landmark/language/history core mechanism |
| E | same as D | multitrajectory | disabled | stop failure decomposition |

Training for A may use `--no_heatmap_trajectory` as a clean ablation, but it must be a matched-start comparison. B/C/D should use the **same checkpoint** and differ only in inference flags. No test_unseen during tuning.

Example short run (not a performance claim):

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=10 BATCH=8 GRID=7 RUN_NAME=heatmap_joint_10e \
  bash multiagent/train_1gpu.sh --trajectory_goal_k 5
```

For B: `--trajectory_use_for_control --trajectory_selector_mode prior`; C: `--trajectory_use_for_control --trajectory_selector_mode joint --no_trajectory_relation_selector`; D: `--trajectory_use_for_control --trajectory_selector_mode joint`; E: add `--trajectory_disable_learned_stop`. Use a local eval entrypoint and `--checkpoint` to reuse the *same* checkpoint. Do not retrain merely to switch inference selector. Since relation/no-relation training itself changes optimization, additionally train matched C/D ablations independently to establish component attribution.

## Measurements

In addition to SR/SPL/OSR/NE and goal TopK coverage:
- `trajectory_prior_hit@20`, `trajectory_joint_hit@20`, `trajectory_oracle_hit@20` (oracle = GT-nearest candidate **coverage diagnostic** only).
- Selected trajectory `prior_FDE_m` / `joint_FDE_m`, minimum ADE/FDE among all proposals.
- `trajectory_rank_supervision_rate`: fraction of evaluated steps with a GT-nearby predicted candidate; essential to assess training signal sparsity.
- `trajectory_stop_precision` and `trajectory_stop_count`: precision among stops actually triggered during evaluated rollout, **not** a substitute for recall/false-stop analysis.
- Evaluate paired per-episode improvements across val_seen / val_unseen and compute bootstrap CIs; record route lengths and action budgets.
- Never compare the GT-using oracle as a deployable method.

## Known issues / next work

- The selector now has separate named-landmark relation tokens with candidate-to-instruction attention and causal pose-history features; it still has **no independently verified landmark appearance or visual correspondence**.
- The current relation head can overfit recurring map layouts or landmark names. Use matched seen/unseen episode-level evaluation, named-reference permutation controls, and hard-negative error analysis.
- Candidate imitation uses the demonstrated future route as proxy even if student deviates from teacher state; off-policy recovery requires dedicated labels.
- No dynamic map collision checks or uncertainty calibration are implemented.
- Learned Stop Head remains weakly supervised; OSR-SR gaps and short path execution require closed-loop validation.
- Do not claim SR/SPL gains until actual matched CityNav experiments and hard-negative analysis are completed.
