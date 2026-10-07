# Evidence-backed referenced-landmark Top-K reranking

This experiment starts from `experiment/heatmap-system-optimization-20e` and
adds only the strongest already-observed static cue: the
instruction-referenced landmark mask.

## Why this is intentionally small

Prior diagnostics showed that referenced-landmark structure has much stronger
coarse localization value than generic directional relations, trajectory
frames, probabilistic relation programs, or explicit layout reasoning.  This
branch therefore does **not** add those failed/unstable components.

The existing 4-channel dense spatial belief already receives the referenced
mask.  This change adds a deterministic inference-time use of the same legal
input so an existing 20-epoch checkpoint can be evaluated without adding model
parameters or retraining. The reranker is forcibly disabled for every training
rollout, including the student rollout, so it cannot change the trajectories
used for optimization.

## Reranking rule

1. Run the existing dense 28x28 heatmap and greedy NMS.
2. Keep only the first 4 NMS hypotheses for optional reranking.
3. Downsample the referenced-landmark mask to 28x28 and expand it locally with
   geometric decay.  This creates a proximity prior rather than forcing the
   goal onto the landmark footprint.
4. Combine heatmap log-probability with the reference prior.
5. Replace raw Top-1 only when:
   - raw Top-1 is ambiguous;
   - the replacement has materially stronger reference support; and
   - the replacement already exists in the raw Top-4.

Confident raw Top-1 predictions are preserved.  No target id, target position,
candidate rank label, or GT information enters the reranker.

## Default settings

- Top-K allowed to rerank: 4
- reference weight: 0.50
- maximum raw Top1/runner-up log margin: 0.20
- minimum reference-prior gain: 0.30
- proximity expansion: 1 dense cell
- per-cell decay: 0.70

Use `--disable_reference_rerank` for the exact control behavior of the base
branch.

## Diagnostics

The existing HEATMAP_DIAGNOSTICS line now also reports:

- `ref_rerank_changes`
- `ref_rerank_rescues`
- `ref_rerank_regressions`

Rescue/regression use GT only after the decision for offline evaluation.  They
never participate in candidate selection.

Do **not** retrain to evaluate this post-process. The first closed-loop comparison should load the same already-trained `experiment/heatmap-system-optimization-20e` checkpoint and use the same seed:

- base branch / or `--disable_reference_rerank`
- this branch with default reranking

Compare SR, OSR, SPL, NE and the rescue/regression counts before considering
any additional relation or visual modules.
