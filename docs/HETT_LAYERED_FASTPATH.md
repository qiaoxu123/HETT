# HETT Layered Fastpath: Static / Dynamic / Planning (2026-10-09)

Branch: `2027-CVPR/hett-layered-fastpath` (based on `2027-CVPR/heatmap-joint-goal-trajectory-fixes` at `c385a913`).

## Goal and guardrails

Remove repeated computations **without** changing the HETT candidate scores, predicted trajectories, teacher/student training objectives, or route actions in the default experiment. Performance improvements are hypotheses until a matched local 5090 GPU benchmark. This branch does NOT port SBFNav selector weights, discard student rollout, or replace Darknet/SigLIP.

| Layer | Changed computation | Expected benefit | Model behavior |
| --- | --- | --- | --- |
| Static map | Global and referenced contour raster channels computed once per episode map object | fewer NumPy copies/conversions every step | exactly same four input channels |
| Static named anchors | Normalize referenced contours and aggregate centroids once per episode reset; preserve name/contour order | removes repeated contour loops for every UAV move | same `center_xy` / `extent_xy`, no extra oracle data |
| Static language | Bounded LRU token IDs for stable reference names | less duplicate tokenization in teacher/student rollouts | same token IDs for fixed tokenizer |
| Dynamic observation | Preserve fresh view/explored masks, UAV poses, and RGB crops at every step | correctness: only static data cached | full online feedback still used |
| Planning | Share sampled goal features between `_generate` and relation selector; calculate identical token projection only once per step | fewer GPU operations | same autograd graph and predictions |
| Training diagnostics | Skip **no-grad** oracle path ADE/FDE diagnostic loop during training unless `--trajectory_train_diagnostics` | less synchronization and unnecessary work | training losses, positive candidate masks and action decisions remain |
| Fast evaluation (opt-in) | `fast_eval=true`: skip GT heatmap, oracle paths and read-only per-step observer while still computing official end-path metrics | faster frequent validation | SR/SPL/OSR/NE still use executed paths; ADE/FDE and selector diagnostics N/A |
| Profiling (opt-in) | `profile_rollout=true` measures CUDA-synchronized: reset, RGB+Darknet, ET+Belief+Planner, loss/control, dynamic obs and backward | identify actual bottlenecks | incurs additional synchronization; never use as accelerated throughput result |

### Important known remaining bottlenecks

- The active `train(feedback='student')` still executes **both teacher and student closed-loop rollouts** per batch. Removing this changes the training distribution and is intentionally not done.
- The RGB/Darknet and BERT pipelines are still active (legacy ET auxiliary heads remain supervised). No pretrained visual feature cache is enabled because the backbone is trainable; silently caching it would change gradients.
- The architecture is still RGB-blind inside the new joint goal/trajectory heads. This fastpath is not a new RGB grounding method.
- Nontrivial architectural options (Top-K -> Top-5 conditional trajectories, frozen-backbone warmup, long-term visual memory, continuous control) require separate accuracy ablations, not silent edits here.
- Static descriptors are specific to the active episode batch and are replaced every time `_get_obs(poses=None)` resets; do not reuse across different maps/goals without a key.

## Recommended local verification (GPU on CityNav host)

1. Set the checkpoint and base-arguments environment variables:
```bash
export HETT_INITIAL_CHECKPOINT=/absolute/path/to/common_initial.pt
export HETT_BASE_ARGUMENTS=/absolute/path/to/base_arguments.json
```
2. Run CPU regressions (GitHub Actions also runs these):
```bash
python -m pytest -q tests/test_layered_fastpath.py tests/test_candidate_relation_selector.py tests/test_heatmap_trajectory.py tests/test_multi_landmark_relations.py
```
3. Run matching smoke tests using clean output directories:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_goal_trajectory_experiment.py \
  --config configs/experiments/hett_layered_fastpath_6e.json \
  --output artifacts/layered_fastpath_smoke --smoke
```
This script already runs two optimizer steps at B=2 and B=8, checks parameter update norms and evaluates short validation episodes. It is not a reliable convergence result.

4. For phase attribution, copy the JSON experiment config and set `profile_rollout=true`; run the same smoke test in another clean directory. The output `gpu_batch*.json` includes `profile_seconds`. **Do not compare its wall-time to the non-profiled run** because profiling synchronizes the GPU after each stage.
5. For a throughput comparison, compare non-profiled B=8 `seconds_per_batch`, peak VRAM and real `train_seconds` against the original fix-branch commit under identical data, seed, checkpoint, hardware, PyTorch version and rollout length. Record the chosen commit SHA from `manifest.json`.
6. Compare old and new non-profiled smoke directories with the included safety-checked script:
```bash
python scripts/compare_layered_fastpath.py \
  --before artifacts/ancestor_smoke \
  --after artifacts/layered_fastpath_smoke \
  --output artifacts/LAYERED_FASTPATH_COMPARISON.md
```
The tool requires the **same initial checkpoint SHA-256** and identical optimizer-step counts. The ancestor must be the pre-fastpath fixes commit; using only the optimized branch on both sides proves nothing.

7. Run full Seen/Unseen validation with default `fast_eval=false` before claiming metric equivalence. For quicker official-metric-only checks, set `fast_eval=true` (the rank/ADE/FDE observer metrics will be missing on purpose).
8. Do not run another 6 epochs or claim SBFNav-equivalent training speed until equivalence, run time, and VRAM have been measured.

## Rollback / ablation

- Full diagnostic training: set `trajectory_train_diagnostics=true` in the JSON config.
- Full diagnostic validation: `fast_eval=false`.
- A rigorous before/after speed comparison uses the ancestor commit on a separate clean checkout with the same baseline checkpoint and config.
- If the head checkpoint or validation metrics differ beyond expected stochastic tolerance, first compare static reference descriptors and shared spatial/attention projections with the CPU tests and investigate code semantics.

## What to report

`train_seconds`, `profile_backward_seconds`, `profile_visual_preprocess_and_darknet_seconds`, `profile_et_belief_and_planning_seconds`, `profile_dynamic_observation_seconds`, `profile_loss_diagnostics_and_control_seconds`, time/batch, peak allocated/reserved VRAM, SR/SPL/OSR/NE, initial Top-20 Hit@20 and Goal selection Hit@20. Only **full diagnostic mode** provides all ranking/trajectory metrics.
