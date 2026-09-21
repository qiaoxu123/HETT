# RSRefSeg2-inspired HETT grounding handoff

## Scope and current state

This branch is the planned successor to `codex/landmark-teacher-cleanup`. At the
handoff point, the RSRefSeg2-inspired HETT model change has **not** been implemented or
trained. Only the single-seed protocol is frozen in `PROTOCOL.md`. Do not report this
branch as an experimental result.

The parent teacher-cleanup implementation is complete and committed. Its full
three-epoch run was still in epoch 1 at the handoff snapshot, so it has no navigation
conclusion yet. Finish or reproduce that parent experiment before starting this one.

## Authoritative commits and worktrees

- HETT baseline/main: `51a1828`.
- Landmark arrival audit: `361445d`.
- Landmark-aware teacher cleanup implementation: `65ac5d8`.
- Teacher-cleanup reproduction scripts and documentation: `3115b9f`.
- Teacher-cleanup in-progress snapshot: `3d09c89`.
- Standalone RSRefSeg2 CityRefer fine-tuning: `98c267a` on
  `codex/finetune-rsrefseg2-cityrefer`.
- This planned model experiment: `codex/rsrefseg2-goal-grounding`.

Run artifacts are ignored by Git and must be transferred separately if the exact local
checkpoints are required. Dataset and weight symlinks are machine-specific and must not
be committed.

A portable source-history bundle was generated at
`/home/tenant2/Workspace/hett-rsrefseg2-handoff-20260921.bundle`; its adjacent
`.sha256` file records the checksum. It contains `main`, the teacher-cleanup and
planned grounding branches, and both standalone RSRefSeg2 branches. On another machine:

```bash
git clone /path/to/hett-rsrefseg2-handoff-20260921.bundle hett-crotonyl
cd hett-crotonyl
git switch codex/rsrefseg2-goal-grounding
```

## Findings already supported by evidence

### HETT baseline and stopping diagnosis

- Fixed 20-epoch seed-0 baseline best checkpoint: `val_seen` SR 31.78 / SPL 24.13;
  `val_unseen` SR 18.91 / SPL 14.73.
- Replacing only the learned stop decision with privileged ground-truth stopping inside
  20 m changed results to `val_seen` SR 54.94 / SPL 52.01 and `val_unseen` SR 34.93 /
  SPL 32.72.
- The paired trajectories are identical before stopping. Therefore stopping is the
  largest demonstrated bottleneck; this does not prove that goal grounding is already
  sufficient.

### Landmark/teacher diagnosis

- On full `val_unseen`, trajectories enter a supplied landmark's 20 m region in 96.96%
  of episodes, but only 80.77% are still inside at the stage transition and 11.17%
  leave after arrival.
- The teacher-cleanup implementation uses two consecutive samples inside 20 m as stable
  arrival, splits coarse navigation from local exploration, fixes batch-shared stage
  state, preserves human yaw, and leaves validation references unchanged.
- Offline `train_seen` audit: 17,774 / 21,878 routes (81.24%) can be stably split;
  original median 40 points becomes 12, while endpoint error remains 0.0 m.
- The full three-epoch run had reached 72.4% of epoch 1 at 2026-09-21 22:03 +08:00.
  No epoch checkpoint or validation metric existed, so no SR/SPL improvement claim is
  valid at handoff time.

### Standalone RSRefSeg2 CityRefer result

RSRefSeg2 was fine-tuned for two epochs on a map-balanced 4,096-example subset of
`train_seen` using target-plus-landmark local regions. On the frozen `test_unseen`
evaluation used in that completed experiment, zero-shot to fine-tuned changes were:

| metric | zero-shot | fine-tuned |
|---|---:|---:|
| mean box IoU | 3.17 | 18.34 |
| Acc@0.25 | 1.38 | 29.98 |
| Acc@0.50 | 0.11 | 8.46 |
| center accuracy | 20.66 | 33.21 |
| mask gIoU | 1.53 | 13.95 |

Paired same-image instruction switching (192 pairs) improved alternate-target
Acc@0.25 from 0.52 to 17.19 and reduced prediction-mask overlap from 68.08 to 50.06,
but both-target correctness at 0.25 was only 8.85. Fine-tuning therefore adds real
language sensitivity but remains too weak to serve as a reliable navigation endpoint
detector by itself.

The local-region breakdown also shows scale sensitivity: Acc@0.25 was 31.07 for ROI
sizes up to 160 pixels, 16.92 for 160--240, and 6.17 above 240. Direct target crops
outperformed broader context crops (33.98 versus 27.53 Acc@0.25).

### Earlier HETT grounding experiments

- A relation heatmap using landmark identities and contours reached `val_unseen`
  Top-1 Hit@20 27.73 and Top-5 Recall@20 58.00.
- Factorized direction-by-distance bases reached 27.81 / 57.20. Their privileged
  candidate-set oracle was 91.32% Hit@20, showing that candidate coverage is adequate.
- Token-level language-to-landmark binding overfit seen cities: `val_seen` Top-1 44.33
  but `val_unseen` 23.68.

The next model must therefore improve language-conditioned candidate selection and
calibrated stopping, rather than only enlarging the geometric candidate set.

## Required experiment on the next machine

1. Reproduce or finish the three-epoch parent teacher-cleanup run and record paired
   `val_seen` / `val_unseen` metrics. Never use `test_unseen` during development.
2. Preserve the map encoder's 15 x 15 spatial grid instead of flattening it before goal
   prediction.
3. Add a language-conditioned coarse heatmap and a bounded within-cell offset head.
4. First evaluate goal Hit@10/20, median/P90 error, and same-map instruction-switch
   sensitivity without changing controller stopping.
5. Only if unseen grounding improves, add an independent semantic stop/arrival head and
   evaluate stop precision/recall, premature-stop rate, timeout rate, SR, and SPL.
6. Keep diagnostic code independent from stop control flow and add an AST/runtime test
   so logging cannot short-circuit termination.
7. Use seed 0 for screening. A candidate requires about +3 points unseen SR with no
   material SPL regression; one seed is not a publication claim.

## Reproduction constraints

- Use the target machine's supported PyTorch/CUDA build; do not blindly recreate the
  old README environment.
- Use one GPU task at a time and verify with `nvidia-smi` before launch.
- Store multi-gigabyte checkpoints on a data volume and keep only symlinks in the
  worktree.
- Start experiments through `scripts/supervise_experiment.py` so source snapshots,
  commands, hashes, status, and telemetry are retained.
- Treat `data/` and `weights/` as read-only.
