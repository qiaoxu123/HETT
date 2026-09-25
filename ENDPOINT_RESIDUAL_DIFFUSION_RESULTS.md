# Endpoint-conditioned residual trajectory diffusion V2

## Change

The frozen V1 remains untouched.  V2 adds the belief-hypothesis endpoint,
endpoint displacement, direction, minimum required steps, and remaining budget
to the diffusion condition.  It diffuses only lateral offsets around a straight
shortest-path skeleton and predicts zero to two extra exploration steps.

Ground-truth target coordinates are not model inputs or trajectory supervision.
They are used only for final metrics.

## Generation gate

Training used the first 1,024 `train_seen` episodes; the following 256 episodes
were held out.  On the held-out slice:

| Metric | V2 result |
|---|---:|
| Endpoint error, mean | approximately 0 m |
| Endpoint error, maximum | 0.0000023 m |
| Path / direct-distance ratio, median | 1.0000 |
| Path / direct-distance ratio, mean | 1.0112 |
| Paths longer than 110% | 1.07% |
| Reverse-progress steps | 0.028% |
| Mean extra steps | 0.40 |
| Paths over 20 steps | 0 |
| Best-of-8 Hit@20 | 49.22% |

The generator therefore passed endpoint, efficiency, coverage, and budget
checks.

## Strict 256-episode closed-loop gate

The same first 256 `val_seen` episodes and frozen 20-action protocol were used.

| Method | SR | SPL | NE | Oracle SR |
|---|---:|---:|---:|---:|
| Frozen V1, stable commit | **28.52%** | **26.15%** | 44.01 m | 34.77% |
| V2, original full scorer | 23.05% | 21.68% | **39.05 m** | 30.47% |
| V2, goal-only scorer diagnostic | 23.83% | 22.50% | **39.03 m** | 32.03% |

V2 improves average final distance and path regularity but fails the required
SR/SPL non-regression gate.  Its direct slots have approximately zero lateral
deviation.  Its three observation slots learn about 8--10 m mean peak lateral
deviation and roughly 1.03 path/direct ratios, whereas the corresponding rule
teachers use about 13--14 m and 1.17--1.21 ratios.  Removing those intentional
detours reduces the chance of observing or passing the true target when the
selected endpoint is wrong.

## Decision

Keep V2 as the efficient endpoint-conditioned generator prototype, but do not
replace the frozen V1 closed-loop model yet.  The next conceptual step would be
to restore only relation- and uncertainty-justified observation offsets while
retaining V2 for direct routes.  Per user instruction, that next step was not
implemented in this experiment.

No `val_unseen` or `test_unseen` data were loaded for V2 development.
