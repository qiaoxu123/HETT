# HETT Heatmap -> Multimodal Trajectory Belief (Phase 1)

## Research scope
This branch builds **only on HETT's dense 28x28 heatmap and existing reference-landmark geometry**. The task is to improve CityNav navigation with learned path hypotheses. SBFNav is **only a downstream performance reference**, not the planning architecture or training dependency.

Base: `2027-CVPR/heatmap-sbf-relative-geometry-top20` (itself a HETT Heatmap variant).
New implementation:
- `multiagent/models/heatmap_trajectory.py`: multimodal path anchors, residual correction, learned mode logit, stop head, arclength-resampled teacher supervision, WTA imitation and BCE stop loss.
- `multiagent/models/ET_haa.py`: consumes HETT `SpatialBeliefOutput.spatial_features` and `probabilities`, returns trajectory tensors.
- `multiagent/agent.py`: student proposals derive **exclusively from predicted Heatmap**; a separate teacher-conditioned proposal uses the GT endpoint ONLY in training for a supervised auxiliary loss. GT teacher suffix and Stop labels are never model inputs during evaluation.
- `multiagent/main.py`: logs best-of-K minADE/minFDE in meters, in addition to existing Top-5/20 and navigation SR/SPL/NE.
- `tests/test_heatmap_trajectory.py`: unit tests for endpoint convention, multimodal separation, gradient flow, teacher resampling and shape checks.

## Model

```text
HETT Map + Language + Reference Landmarks + Current Pose
                      |
                 28x28 Heatmap
                      |
            NMS Top-5 goal candidates
                      |
       Spatial-feature-conditioned MLP
                      |
          For each goal: 3 anchor modes
            (left-bend/straight/right-bend)
                      |
        Residual correction -> 8 waypoints
                      |
             joint logits = log P(goal)
                           + log P(mode|goal)
                      |
          15 predicted trajectory proposals
          + independently trained Stop logit
```

The 3 curve anchors are **kinematic hypotheses, not proven obstacle-free paths**. CityNav contour priors alone are not a validated collision map.

## Label generation
Source: `Episode.teacher_trajectory` made from CityFlight/MTurk paths. For each active step:
1. Locate nearest teacher pose to current training pose (label generation only).
2. Take future teacher suffix, append the CityRefer GT goal (explicitly synthetic segment; source human endpoint can be up to ~30m away).
3. Arc-length sample the remaining full-goal route to 8 waypoints.
4. Condition *training-only* trajectory proposals on the GT goal to produce an imitation signal.
5. Teacher imitation: best-matching mode (WTA) + weighted mode selection; Stop positive only near GT **and** near the end of remaining teacher route.
6. Predicted-goal trajectories are separately evaluated against the teacher suffix but never receive oracle endpoints during inference.

When teacher path states and a student rollout disagree, these are proxy labels. Valid closed-loop policy learning should use recovery trajectories in a later stage.

## Run on a trusted GPU environment

```bash
git fetch origin
git switch 2027-CVPR/hett-heatmap-multitrajectory
cd multiagent
CUDA_VISIBLE_DEVICES=0 EPOCHS=1 BATCH=2 GRID=7 RUN_NAME=traj_smoke \
  bash train_1gpu.sh --benchmark_batches 2
```

Do not assume pretrained weights or CityNav data exist in the CI checkout; inspect the paths in `multiagent/defaultpaths.py` first. Save independent checkpoints for each ablation.

Next, for a small real-data experiment (e.g. 10 epochs with fixed seed):
```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=10 BATCH=8 GRID=7 RUN_NAME=heatmap_traj_10e \
  bash train_1gpu.sh
```

Recommended ablation groups: (A) heatmap baseline `--no_heatmap_trajectory`, (B) learned trajectories with legacy controller (default), (C) optional controller `--trajectory_use_for_control`. Do **not** enable option (C) until the trained checkpoint shows reasonable ADE and Stop precision; the mode is provided for future closed-loop experiments. Existing `--no_heatmap_multi_landmark` ablation is independent.

Use the same train/val splits, sampling, seed, backbone weights, map size, success radius, NMS and epoch/step budget in every group. Report:
- Goal coverage@5 and coverage@20 at 20m (step-wise; do not confuse with episode-level recall).
- minADE/minFDE@15 in meters; oracle-best across predictions is diagnostic, not a deployable selector.
- Closed-loop SR, SPL, NE; early-stop rate and path length ratio.
- Model inference latency, GPU memory and epoch duration.
- Oracle Goal + straight-line controller ceiling, to distinguish localization failure from trajectory planner failure.

## Testing and CI
This experiment branch has an automatic GitHub-hosted CPU test workflow:
`/.github/workflows/heatmap-multilandmark-tests.yml`.

Your self-hosted `hett-rental-5001` runner only accepts the owner-approved `HETT self-hosted CI` workflow on `main`. Manual dispatch:
1. Open https://github.com/qiaoxu123/HETT/actions/workflows/self-hosted-ci.yml
2. Run workflow **on main**; set `ref=2027-CVPR/hett-heatmap-multitrajectory`.
3. Check `gpu_check` and enable `run_tests` if the runner's CityNav venv has test dependencies.

The existing self-hosted workflow performs checkout, compile, CUDA status and optional pytest. It **does not run full model forward, training, or data scans**. GPU smoke and real navigation evaluations above must be initiated separately from a trusted session. No untrusted PR execution or changes to the user's runner safety gate are introduced.

## Known limits

- Trajectory planner consumes the current HETT belief and pose; it does not yet use dedicated first-person history features or obstacle height. This is the minimal baseline to isolate whether paths help.
- Stop labels are weakly supervised from distances to GT / remaining teacher route. Do not equate their training accuracy with navigation stop reliability.
- By default the legacy navigation controller remains in use and only emits/logs trajectory proposals; **no SR improvement is implied without selecting and executing them**.
- Multi-goal selection from a heatmap can still fail if the correct goal is absent from Top-5. Study this upper bound before scaling the generator.
- This branch is experimental and must not be merged into develop before an end-to-end dataset-backed training run and full validation.
