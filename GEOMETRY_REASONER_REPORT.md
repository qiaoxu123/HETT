# CityNav Pure Geometry Reasoner Report

## Conclusion

The deployable rule engine gives only a small improvement over Static B0 and
does not meet the 40% R@1/20 success threshold. On val_unseen, R@1/20 rises from
21.43% to 21.99% and R@4/20 from 46.98% to 48.05%. Dynamic teacher pose does not
improve this result. The outcome is **CASE B**: the privileged Oracle is strong,
but explicit-language rules are much weaker.

The main reason is now clear: the former 69.74% Oracle used the true target's
distance and bearing relative to anchors. Instructions usually do not specify
those continuous values. In the strict parser-grounded Oracle, bearing signature
is the dominant privileged signal; executable left/right/near/ordinal/sequence
rules do not recover it.

## Protocol

- No RGB, controller, action training, or navigation rollout.
- Only train_seen, val_seen, and val_unseen were read; test_unseen was forbidden.
- B0 Top-16 candidates are fixed. Goal coordinates are used only for metrics and
  explicitly labelled Oracle construction, never as reasoner input.
- All weights are selected on val_seen and frozen for val_unseen.
- The engine has zero learned parameters.

## Parser coverage

| Split | Instructions | With constraint | With anchor |
|---|---:|---:|---:|
| train_seen | 21,878 | 19,532 (89.3%) | 21,444 (98.0%) |
| val_seen | 2,470 | 2,218 (89.8%) | 2,437 (98.7%) |
| val_unseen | 2,697 | 2,391 (88.7%) | 2,656 (98.5%) |

val_unseen relation counts: near 685, front 561, right 520, left 475,
between 469, behind 458, ordinal 214, nearest 41, farthest 12, before 8,
after 9, and past 5. Across/opposite/along are parsed but marked unsupported
instead of receiving invented geometry.

## Main ablation — val_unseen

| Method | R@1/20 | R@4/20 | R@8/20 | R@1/40 | Top1 Dist | GT Rank |
|---|---:|---:|---:|---:|---:|---:|
| G0 Static B0 | 21.43% | 46.98% | 65.96% | 52.02% | 49.20m | 6.03 |
| G1 + Anchor Distance | **21.99%** | **48.05%** | 65.93% | 51.98% | **47.94m** | **5.87** |
| G2 + Left/Right | 21.43% | 46.98% | 65.96% | 52.02% | 49.20m | 6.03 |
| G3 + Ordinal | 21.39% | 46.94% | 65.78% | 51.98% | 49.30m | 6.04 |
| G4 + Sequence | 21.43% | 46.98% | 65.96% | 52.02% | 49.20m | 6.03 |
| G5 + All language geometry | 21.65% | 47.16% | 65.74% | 51.72% | 49.13m | 5.99 |
| G6 Geometry only | 16.02% | 40.71% | 59.07% | 42.97% | 60.32m | 6.41 |
| G7 B0 + geometry + start pose | **21.99%** | **48.05%** | 65.93% | 51.98% | **47.94m** | **5.87** |

The best gain is +0.56 R@1 and +1.08 R@4. Geometry alone is substantially
worse than B0, so B0 remains essential.

## Single relations

val_seen tuning selected zero weight for direction, near, ordinal, and sequence.
Only between selected a non-zero weight (0.25), reaching 21.88% R@1 and 47.50%
R@4 on unseen: +0.45 and +0.52 points over B0. Valid-episode ratios are 57.4%
direction, 26.5% near, 7.5% ordinal, 0.7% sequence, and 12.6% between.

## Static versus dynamic geometry

| Teacher progress | R@1/20 | R@4/20 |
|---|---:|---:|
| Start | 21.99% | 48.05% |
| 25% | 22.06% | 47.91% |
| 50% | 22.06% | 48.13% |
| 75% | 22.02% | 47.94% |
| 100% | 21.80% | 47.68% |

Current pose is not required: dynamic geometry is flat and slightly worse at
the endpoint. This does not support trajectory-aware geometry belief yet.

## Referenced-anchor sanity check

| Anchor | R@1/20 | R@4/20 |
|---|---:|---:|
| True referenced anchor | **21.99%** | **48.05%** |
| Same-map shuffled | 21.54% | 46.87% |
| Same-map random | 21.73% | 46.50% |
| No anchor | 21.43% | 46.98% |

The true anchor contains real information, but its deployable gain is small.

## Oracle decomposition

Oracle metrics are conditional on episodes where the corresponding relation is
valid. They use the goal only to construct the perfect signature and are not
deployable inputs.

| Oracle signal | Episodes | R@1/20 | R@4/20 |
|---|---:|---:|---:|
| Target anchor-distance signature | 2,644 | 30.79% | 70.31% |
| Target bearing signature | 2,644 | 50.57% | 74.74% |
| Distance + bearing signature | 2,644 | **62.82%** | **80.11%** |
| Perfect direction relation | 1,547 | 34.13% | 72.07% |
| Perfect near relation | 714 | 30.67% | 68.91% |
| Perfect between relation | 341 | 33.72% | 66.28% |
| Perfect ordinal relation | 203 | 18.23% | 55.17% |
| Perfect sequence relation | 19 | 21.05% | 68.42% |

Thus the old 69.74% result mainly came from privileged target bearing plus
distance signatures, not from left/right/near words alone. The strict Oracle is
62.82%, while the actual rule engine is 21.99%: a 40.83-point R@1 gap.

## Candidate-pool size

| Pool | R@1/20 | R@4/20 |
|---|---:|---:|
| Top-4 | 21.99% | 46.98% |
| Top-8 | 22.02% | 48.13% |
| Top-16 | 21.99% | 48.05% |

Geometry is marginally most useful inside Top-8, but the differences are tiny.

## Goal-to-anchor distance

| Distance | Episodes | R@1/20 | R@4/20 |
|---|---:|---:|---:|
| 0–20m | 475 | 49.68% | 81.26% |
| 20–40m | 703 | 22.90% | 59.46% |
| 40–80m | 938 | 14.61% | 36.99% |
| 80m+ | 528 | 10.04% | 23.11% |

Most apparent geometry value is concentrated where the goal is already close
to the referenced anchor.

## Failure taxonomy — val_unseen

| Outcome | Count | Share |
|---|---:|---:|
| Candidate pool miss | 524 | 19.4% |
| Parser has no constraint | 244 | 9.0% |
| Anchor unresolved | 37 | 1.4% |
| Multi-rule conflict | 377 | 14.0% |
| Remaining ranking error | 580 | 21.5% |
| Top-4 success | 935 | 34.7% |

This separates candidate-pool limitations from language and rule failures.

## Debug visualizations and runtime

The debug run exports 100 val_seen and 100 val_unseen top-down PNG/JSON pairs,
including instruction, parsed program, every anchor match, B0 score, individual
geometry components, candidate order, reference frame, final score, start pose,
and GT marker. Artifacts live under
`/mnt/windows-data/hett-geometry-reasoner/debug_viz_s0_20261004_r3/artifacts/`.

Full evaluation takes about 84 seconds on CPU. Model parameters: 0. No GPU was
used.

## Answers

- **A. Significant Top-1/Top-4 gain?** No; +0.56/+1.08 points only.
- **B. Most important executable relation?** Anchor distance, followed by a
  small between gain. Other relation weights collapse to zero on val_seen.
- **C. Is current pose necessary?** No; dynamic results are flat.
- **D. Is the referenced landmark real information?** Yes, verified by shuffle,
  but the usable effect is small.
- **E. Source of the old Oracle?** Mostly privileged target bearing, then target
  anchor distance—not explicit relation words.
- **F. Rule-to-Oracle gap?** 40.83 R@1 points against the strict combined
  signature Oracle.
- **G. Learned geometry matcher?** Not yet. The missing input is largely an
  unspecified target bearing/distance signature; an MLP cannot legitimately
  infer information absent from the instruction. First improve relation/anchor
  clause binding and ordering semantics.
- **H. Add RGB now?** No. Geometry gain is too small and previous RGB evidence
  is below chance. Do not combine two weak signals yet.

## Decision

**CASE B: Oracle strong, rule implementation/representation weak.** Improve
parser clause binding, road-axis extraction, target-vs-reference role assignment,
and ordinal scope before considering a sub-2M learned geometry model. Do not
connect this reasoner to the controller.
