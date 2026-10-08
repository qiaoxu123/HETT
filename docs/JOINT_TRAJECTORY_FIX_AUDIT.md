# Joint Goal/Trajectory Fix Audit (2026-10-09)

Base: `ee9fe9627aaed85512e5dc51ab0ea1204b314c61`

## Verified source-level defects

1. Flat `argmax(goal + mode)` can select a *lower-probability goal* because a sharp conditional mode score outcompetes a diffuse mode distribution for the higher-probability goal. Use marginal goal score `logsumexp` then best mode within selected goal.
2. Original `stop_target` required a location within half the official success radius **and** a teacher-suffix proxy criterion. This is not the actual end-position metric. Now uses the official radius for labels.
3. Inference `stop` was conditioned on Top-1 proposal distance despite joint selection possibly choosing a different goal. It now follows the configured goal selection.
4. With an eight-waypoint plan, the first waypoint may be inside the discrete lookahead controller's 5m STOP radius. A STOP action performs no displacement; this was not distinguished from arrival. Now skip reached points, and track trajectory stagnation.
5. Trajectory path execution did not explicitly mark horizon timeouts, and evaluation's inferred termination incorrectly treated non-horizon terminations as learned stops. Now use actual stop reasons.
6. Training student rollouts defaulted to the legacy two-stage policy whereas validation executed the joint trajectory controller; the experiment runner and continuation runner now apply the selected evaluation variant during student training.

## Validation status

- New CPU regression file: `tests/test_joint_trajectory_selection.py` checks goal/mode marginal selection and stop target radius.
- GitHub Actions CPU workflow amended to include the fix branch and the new tests.
- No new CityNav GPU smoke or end-to-end evaluation has been performed by this audit. Do not assert model accuracy recovery based on code changes.

## Practical follow-up

1. Run the updated GitHub Actions suite, inspect failures and correct regressions.
2. On the local CityNav machine, run the GPU smoke test using a **new output directory**, then evaluate the existing original checkpoint with a fixed controller as a *diagnostic only* (not a new trained policy).
3. Retrain a fresh epoch from the common baseline (old checkpoint is not an appropriate continuation), compare matched train/evaluation results on val_seen and val_unseen, and only continue further epochs after passing the configured gate.
4. Record Stop-positive training rate, selected-goal Hit@20, median path movement, non-progress rate, and per-episode termination reasons. The baseline result of SR ~3% remains the last measured result until new evidence is produced.

Caveat: Original rank supervision labels only the nearest covered candidate. This is not necessarily the only acceptable candidate within 20m. Changing to a soft multi-positive ranking objective is a future experiment, not silently included in this corrective patch.
