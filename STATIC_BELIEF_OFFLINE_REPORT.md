# Static Belief Offline Experiment

Date: 2026-10-04

## Scope

This experiment trains only:

```text
instruction + static landmark raster (+ optional start pose/yaw) -> B0
```

It does not instantiate RGB, depth, trajectory history, explored history, the
navigation environment, Stage 1/2, controller, or rollout. `test_unseen` is rejected
by the loader and was not used.

The branch base and supplied state were verified:

- branch: `2027-CVPR/static-belief-memory`
- supplied HEAD: `c5bbc62f1551e4beef960260100f686f004400e8`
- `849a751` is an ancestor and the supplied HEAD is seven commits ahead

## Representation gate

```text
python -m unittest \
  discover -s tests -v

12 tests passed
```

The added tests also assert that the four ablations use a fixed five-channel
architecture while unavailable channels are exactly zero. The original static
extractor still selects only HETT map channels 2 and 3.

## Protocol

- Train: 21,878 `train_seen` episodes.
- Validation: 2,470 `val_seen`; 2,697 `val_unseen`.
- Frozen local BERT features, compressed into four ordered phrase tokens.
- Static raster: 240 x 240; belief output: 30 x 30 (13.67 m/cell).
- Gaussian target sigma: 20 m; greedy NMS kernel: 3.
- Fixed model: five map channels, hidden size 128, identical initialization per
  seed; disabled inputs are zeroed.
- Optimizer: AdamW, batch 16, learning rate 3e-4, bf16, seed 0 for the complete
  A-D diagnostic.
- C/D were repeated for two epochs with seeds 0, 1, and 2.

The ablations are independently trained under the same protocol. They do not reuse
one trained checkpoint with channels masked only at evaluation time.

### Leakage audit

- Exact/case-insensitive referenced-name coverage is 99.32% on train, 100.00% on
  `val_seen`, and 99.93% on `val_unseen`.
- The final target point lies inside a referenced-landmark mask in only 17.39% of
  train, 15.75% of `val_seen`, and 11.23% of `val_unseen` episodes. The reported
  Top-K result is therefore not explained by simply copying a mask that contains
  the target in most episodes.
- Existing processed annotations list the target object's own name as a referenced
  landmark in 0.37% of train, 0.24% of `val_seen`, and 0.04% of `val_unseen`.
  These rare existing annotations were preserved to match HETT's current input.
- Current-view and explored channels are never loaded by the offline dataset.

## One-epoch A-D diagnostic (seed 0)

### `val_seen`

| Input | Top-1 distance | R@8/20m | R@16/20m | R@8/40m | R@16/40m | OracleDist@8 |
|---|---:|---:|---:|---:|---:|---:|
| A Instruction | 261.4 m | 8.58% | 15.02% | 14.62% | 24.90% | 172.5 m |
| B + Global contour | 158.4 m | 27.21% | 39.72% | 49.72% | 67.49% | 57.3 m |
| C + Referenced mask | 45.6 m | 75.59% | 84.78% | 92.47% | 95.99% | 18.7 m |
| D + Start pose/yaw | 45.5 m | 74.82% | 84.49% | 91.66% | 95.43% | 19.1 m |

### `val_unseen`

| Input | Top-1 distance | R@8/20m | R@16/20m | R@8/40m | R@16/40m | OracleDist@8 |
|---|---:|---:|---:|---:|---:|---:|
| A Instruction | 272.9 m | 3.97% | 8.94% | 7.71% | 14.94% | 187.9 m |
| B + Global contour | 195.1 m | 11.38% | 24.88% | 30.03% | 56.88% | 71.2 m |
| C + Referenced mask | 47.9 m | 66.89% | 80.57% | 88.62% | 93.10% | 20.6 m |
| D + Start pose/yaw | 47.7 m | 67.82% | 80.76% | 88.06% | 92.88% | 20.7 m |

The large B-to-C change is the primary result. Start pose/yaw does not provide a
consistent one-epoch gain.

## Two-epoch C result (mean +/- sample standard deviation, three seeds)

| Split | R@1/20m | R@4/20m | R@8/20m | R@16/20m |
|---|---:|---:|---:|---:|
| `val_seen` | 28.83 +/- 0.14% | 62.40 +/- 0.72% | 77.11 +/- 0.61% | 86.80 +/- 0.65% |
| `val_unseen` | 23.96 +/- 1.31% | 51.92 +/- 2.49% | 69.03 +/- 0.06% | 81.86 +/- 0.53% |

| Split | R@1/40m | R@4/40m | R@8/40m | R@16/40m |
|---|---:|---:|---:|---:|
| `val_seen` | 62.44 +/- 0.89% | 88.29 +/- 0.65% | 94.28 +/- 0.47% | 96.82 +/- 0.20% |
| `val_unseen` | 54.54 +/- 2.26% | 80.00 +/- 1.66% | 89.46 +/- 0.11% | 94.57 +/- 0.33% |

| Split | Top-1 distance | OracleDist@8 | OracleDist@16 | Entropy |
|---|---:|---:|---:|---:|
| `val_seen` | 44.78 +/- 0.39 m | 17.29 +/- 0.46 m | 14.21 +/- 0.23 m | 5.054 +/- 0.188 |
| `val_unseen` | 47.98 +/- 1.30 m | 19.97 +/- 0.13 m | 15.76 +/- 0.22 m | 5.060 +/- 0.218 |

For seed 0, C improves from epoch 1 to 2 on `val_unseen`:

- Recall@8/20m: 66.89% -> 69.04%
- Recall@16/20m: 80.57% -> 82.46%
- Recall@8/40m: 88.62% -> 89.43%
- Recall@16/40m: 93.10% -> 94.96%
- OracleDist@8: 20.62 m -> 19.95 m

This is a useful Top-K prior after only two epochs even though Top-1 remains
ambiguous, matching the intended role of B0.

## Does start pose/yaw help?

At epoch 2, D's three-seed `val_unseen` results are:

- Recall@8/20m: 69.06 +/- 1.24%
- Recall@16/20m: 82.92 +/- 0.71%
- Recall@8/40m: 89.48 +/- 0.28%
- Recall@16/40m: 94.77 +/- 0.21%
- OracleDist@8: 19.98 +/- 0.09 m

These overlap C closely. With the present encoding and two-epoch budget, start
pose/yaw is not necessary for a useful B0 and has no demonstrated material gain.

## Interpretation and limitations

The experiment supports the following claims:

1. Static map structure is necessary for world-referenced localization; instruction
   alone cannot reliably select a world cell.
2. Global contours provide a real but limited prior.
3. The instruction-referenced landmark mask is the dominant signal and reliably
   places the final target within a small Top-K set on both seen and unseen maps.
4. Start pose/yaw is optional at this stage.

This does not yet prove that the model learned arbitrary language-to-geometry
reasoning. The referenced mask is derived from annotations, and much of the gain may
come from the geometric proximity between referenced landmarks and the target. A
future control should shuffle referenced masks within the same map and measure the
drop. It also does not validate dynamic belief updates or navigation success.

## Artifacts

- Full A-D seed-0 run:
  `/mnt/windows-data/hett-static-belief/static_belief_ablation_e1_s0_20261004`
- C/D two-epoch runs:
  `/mnt/windows-data/hett-static-belief/static_belief_cd_e2_s{0,1,2}_20261004`
- Each run contains `provenance.json`, `commands.json`, `status.json`, source
  snapshot, log, checkpoints, cached language features, and `artifacts/metrics.json`.

No navigation training or `test_unseen` evaluation was run.
