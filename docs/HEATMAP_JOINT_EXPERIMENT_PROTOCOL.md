# Joint Goal/Trajectory 1-epoch paired experiment

Remote source commit: `26e1ff0`, branch `2027-CVPR/heatmap-joint-goal-trajectory`.
Read upstream `docs/heatmap_joint_goal_trajectory.md` before execution. Model files and controller decisions are unmodified; `agent.py` adds two optional read-only callbacks for gradient auditing and post-action telemetry. Existing known diagnostic names are not treated as ground truth: Top-K coverage is recomputed from actual candidate IDs, and rank-eligible rate from actual trajectory goal coverage.

## Fixed experiment

- Shared initialization: completed epoch-1 `heatmap_relative_geometry_top20_688e981_b8_s0_2e_20261008/best_val_unseen`.
- Strict load of all existing backbone/head weights. Only newly added `trajectory_head.*` parameters may be missing. Fresh optimizer; common initialization is saved locally as `initial.pt`.
- Seed 0, batch 8, original optimizer/lr/controller settings imported from that run. Full train_seen 21,877 episodes; full val_seen 2,470 / val_unseen 2,697. The runner reads only these three splits.
- 1 **new** training epoch (user revised the budget before training started). Human+student training rollouts follow upstream `feedback=student` behavior. Legacy training control remains the branch default; trajectory objectives train the new heads concurrently. No hyperparameter search.
- Exact config: `configs/experiments/joint_goal_trajectory_1e.json` (K=5 goals, 3 modes, 8 waypoints).
- End-of-epoch full A/B/C/D paired closed-loop evaluation, same model weights across groups. Final comparison is fixed epoch 1, not each group's best epoch. A disables trajectory execution but retains the shared joint-trained backbone; this isolates inference execution/selection rather than separately retraining four models.
- B prior, C joint, D joint without Learned Stop; other action budgets and evaluator settings identical.
- Oracle minima only appear in logging after action choice; inference hook forbids `trajectory_teacher_goal` in eval mode. No model code or structural changes.
- Checkpoints and RNG state retained at every epoch; no implicit resume offset from the old checkpoint.

## Preflight

31 tests passed (28 upstream + 3 experiment contracts), syntax compilation passed.
Real-data GPU smoke used two disposable optimizer batches each at batch 2 and 8, then 8 validation episodes per split for all four modes (8 split-runs). These weights are not used for training initialization.

| Batch | seconds/batch (2-batch smoke) | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|
| 2 | 3.568 | 13.85 | 14.33 |
| 8 | 2.896 | 52.73 | 53.16 |

All losses, gradients and updates finite. Ranking scorer, curve residual, mode scorer and stop head all received nonzero gradients. Zero-initialized heads have near-zero upstream context gradients on the first update and nonzero context gradients on the second, consistent with their initialization. No initialization change was made.

Paired episode IDs, initial Heatmap candidates, episode-recomputed SR/OSR/SPL, Stop confusion-count totals and D's disabled stops were checked. GPU time estimates from two batches are preliminary, not steady-state throughput guarantees. Full run includes 8 split-variant evaluations after the single epoch.

## Outputs and metric definitions

`artifacts/joint_goal_trajectory/run1/`: manifests, split ID lists and annotation hashes, per-epoch training/gradient/memory/timing records, checkpoint hashes, full per-episode traces, JSON/CSV aggregates, paired bootstrap results, report and final figures.

Selected plan ADE/FDE compare policy-state proposals to resampled teacher suffix labels; this is not a collision or recovery optimality score. A's executed plan ADE/FDE is N/A (it executes its original controller); its counterfactual proposed-path diagnostics remain available. Stop is evaluated against the benchmark 20m radius, with precision/recall/accuracy and TP/FP/FN/TN. Switching is candidate-cell ID change per adjacent active decision. Report initial-state and all-step Top-K coverage separately. Path length follows the official macro-state evaluator. Null metrics are not silently converted into zeros.

The bootstrap is paired by episode, seed 0, 10,000 repetitions, with no claim of independence across scenes. Per-epoch peeking is only reporting; final epoch and hyperparameters are fixed.

## Reproduction commands

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/20260922_1/miniconda3/envs/AirVLN39/bin/python -m pytest -q tests
/home/20260922_1/miniconda3/envs/AirVLN39/bin/python -m py_compile multiagent/models/heatmap_trajectory.py multiagent/models/ET_haa.py multiagent/agent.py multiagent/main.py multiagent/parser.py
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/20260922_1/miniconda3/envs/AirVLN39/bin/python scripts/run_joint_goal_trajectory_experiment.py --config "$PWD/configs/experiments/joint_goal_trajectory_1e.json" --output "$PWD/artifacts/joint_goal_trajectory/smoke" --smoke
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/20260922_1/miniconda3/envs/AirVLN39/bin/python scripts/run_joint_goal_trajectory_experiment.py --config "$PWD/configs/experiments/joint_goal_trajectory_1e.json" --output "$PWD/artifacts/joint_goal_trajectory/run1"
```

Use fresh output directories; completed checkpoints and prior results are never overwritten. Raw traces/checkpoints stay on disk; compact results/reports/figures are committed and pushed after successful completion. Push errors are recorded rather than reported as success.

## Budget revision

User requested only one epoch before the formal run launched. The 10-epoch config is retained as an unused template. Active config is `joint_goal_trajectory_1e.json`; the process stops after epoch 1 plus the eight full split-variant evaluations. No automatic continuation to epochs 2–10.
