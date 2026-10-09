# HETT staged backward: retain objectives, release rollout graphs earlier

Baseline: `2027-CVPR/hett-layered-fastpath` at `b184855390fa8c2af4aa66fad8b1c953860cdf36`.
Experiment branch: `2027-CVPR/hett-sequential-backward`.

## Verified bottleneck in source

`NavCMTAgent.train(feedback='student')` originally builds the COMPLETE teacher rollout graph, then the COMPLETE student graph, and only then calls `self.loss.backward()`. With a long dynamic rollout, both activations co-exist. The reported same-checkpoint fastpath GPU smoke was:

| Batch | Seconds/batch | Peak allocated VRAM |
|---:|---:|---:|
| 2 | 4.205 | 13.88 GiB |
| 8 | 3.321 | 52.87 GiB |

These are user-provided smoke outputs, *not* numbers measured by this patch. The layered fastpath at that checkpoint delivered **0.957x / 0.926x**, not a confirmed throughput win over `c385a913`.

## Proposed memory optimization (opt-in, not a model redesign)

Enable `trajectory_sequential_backward=true` (CLI: `--trajectory_sequential_backward`) to execute:

1. `zero_grad()` all three optimizers.
2. Teacher forward: unchanged supervision and on-policy teacher actions.
3. Teacher `backward()`: accumulate gradients in the model parameters; release its large autograd graph.
4. Student forward: unchanged policy actions, supervision, and language/visual models.
5. Student `backward()`: ADD to the parameter gradients from teacher.
6. Run original single gradient clip, **one step per optimizer**.

Total loss for logs is the detached sum. This should preserve the mathematical sum of gradients if teacher/student forward graphs are independent (the source currently constructs independent rollouts), although floating-point accumulation, CUDA kernels, stochastic/dropout differences, and mutable forward buffers can cause small drift. Use the unchanged legacy method by setting the flag to `false`.

**Predicted benefit:** lower peak activation memory, especially at batch=8. It may not speed up an individual batch and adds no GPU parallelism. The same optimizer update count and training dataset remain.

## Local validation before a long training run

Use the same baseline checkpoint and CityNav configuration under comparable GPU load. The layered-fastpath config is the control; the sequential-backward config is the variant.

```bash
git fetch origin
git switch 2027-CVPR/hett-sequential-backward
git pull

export HETT_INITIAL_CHECKPOINT=/absolute/path/to/initial.pt
export HETT_BASE_ARGUMENTS=/absolute/path/to/base_arguments.json

CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_goal_trajectory_experiment.py \
  --config configs/experiments/hett_layered_fastpath_6e.json \
  --output artifacts/control_original_backward --smoke

CUDA_VISIBLE_DEVICES=0 python scripts/run_joint_goal_trajectory_experiment.py \
  --config configs/experiments/hett_sequential_backward_6e.json \
  --output artifacts/sequential_backward --smoke

python scripts/compare_layered_fastpath.py \
  --before artifacts/control_original_backward \
  --after artifacts/sequential_backward \
  --output artifacts/SEQUENTIAL_BACKWARD_COMPARISON.md
```

The smoke script uses just two training optimizer steps at batch 2 and 8 and an 8-episode validation subset. Training runtimes of two such short steps are strongly affected by warm-up and measurement noise. Repeat the paired tests (new directories), alternate order, and do not report a speedup without consistent measurements.

If the machine has <53 GiB of usable single-device VRAM, the original batch=8 baseline cannot be safely reproduced. **Check `nvidia-smi` and available GPU memory first**. Batch=4 may be more realistic on 32-GiB cards; do not infer that 52.87 GiB was measured on a single RTX 5090.

## Tests and failure criteria

CPU unit tests compare parameter gradients and a single AdamW update between joint-sum backward and sequential accumulation under deterministic tiny networks. GitHub Actions runs the full prior regressions plus `tests/test_rollout_backward.py`.

For actual training equality, compare:
- Same checkpoint SHA-256, episode IDs, seed, batch, horizon, GPU, datatype.
- Teacher and student loss components, gradient norms after both rollouts, post-step weights on a deterministic batch.
- Peak allocated GiB and seconds per batch *without* `profile_rollout`.
- Full `val_seen/val_unseen` SR/SPL/OSR/NE before any publication claim.

Do NOT reuse an old 3%-SR checkpoint and do not assume a 2-step smoke shows convergence. If sequential gradients or full eval diverge unexpectedly, disable flag and investigate shared graphs, stateful model buffers, or random-number-generator state.
