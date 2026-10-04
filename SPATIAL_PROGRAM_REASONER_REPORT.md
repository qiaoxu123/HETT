# Spatial Program Reasoner Report

## Executive result

The executable spatial-program representation did **not** improve frozen Static B0. All structured variants P2–P6 selected geometry weight `lambda=0` on `val_seen`. On frozen `val_unseen`, the best nonzero result remained the previous flat geometry baseline: R@1/20 changed from 21.43% to 21.99%, while R@4/20 stayed 46.98%. Forcing the Full Spatial Program to contribute (`lambda=1`) reduced R@1/20 to 17.43% and R@4/20 to 43.75%.

This is a negative result. No RGB, controller, action prediction, navigation rollout, learned geometry model, or `test_unseen` data was used.

## Protocol

- Base: `2027-CVPR/geometry-reasoner@7742ade30325ca9a1effd02e0f2b4bff4f858fb5`.
- Inputs: instruction, instruction-linked map landmark/object geometry, start pose/yaw, and frozen B0 Top-16.
- Tuning: `val_seen` only. `val_unseen` was evaluated after lambda selection.
- GT target coordinates were used only after inference for metrics and debug drawings.
- 0 learned parameters; CPU-only. Frozen P0–P6 evaluation took 170.21 s for 2,470 seen and 2,697 unseen episodes (about 33.0 ms per evaluated episode including seven variants and object graph queries).
- Primary artifact: `runs/spatial_program_eval_s0_20261004_r3/artifacts/metrics.json`.

## Parser census

The parser scanned 21,878 train_seen, 2,470 val_seen, and 2,697 val_unseen instructions (27,045 total development instructions). It emitted at least one constraint for 85.3%, 85.9%, and 85.0%, respectively. On val_unseen, common emitted relations were near (714), front (562), between (493), behind (463), right (337), left (303), and ordinal (214). Explicit sequence words were rare: before 8, after 9, past 5.

Clause counts on val_unseen were 1,735 single-clause, 610 two-clause, and 352 with at least three clauses. The deterministic parser averaged 0.133 ms/instruction.

## Parser benchmark status

The requested 200-record stratified annotation queue is in `analysis/spatial_program_gold.json`. It covers direction, near, between, ordinal, multi-clause, and rare sequence cases. Every record is explicitly marked `human_reviewed=false`.

Therefore a valid Gold parser table and P7 upper bound are **not reported**. The guarded Gold evaluator refuses to run P7 until at least 150 records are independently reviewed. A silver consistency diagnostic is available but must not be interpreted as accuracy:

| Reference | Entity F1 | Role Acc | Relation F1 | Binding Acc | Axis Acc | Ordinal Scope Acc | Exact Match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unreviewed silver queue | 80.44 | 7.00 | 100.00 | 77.45 | 100.00 | 100.00 | 7.00 |

The low role/exact agreement occurs because the stored review candidates include annotation-linked entity names while the diagnostic parser invocation intentionally uses instruction text alone. This reveals sensitivity, not human accuracy.

## Main result (frozen val_unseen)

All metrics are percentages except distance/rank. P2–P6 equal P0 because val_seen chose lambda=0.

| Method | lambda | R@1/20 | R@4/20 | R@8/20 | R@1/40 | Top1 Dist (m) | GT Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 Static B0 | 0.00 | 21.43 | 46.98 | 65.96 | 52.02 | 49.20 | 6.03 |
| P1 Flat Geometry | 0.25 | **21.99** | 46.98 | **66.11** | 51.98 | **48.88** | **6.00** |
| P2 Structured + old frame | 0.00 | 21.43 | 46.98 | 65.96 | 52.02 | 49.20 | 6.03 |
| P3 + Axis Resolution | 0.00 | 21.43 | 46.98 | 65.96 | 52.02 | 49.20 | 6.03 |
| P4 + Ordinal Scope | 0.00 | 21.43 | 46.98 | 65.96 | 52.02 | 49.20 | 6.03 |
| P5 + Object-aware (max) | 0.00 | 21.43 | 46.98 | 65.96 | 52.02 | 49.20 | 6.03 |
| P6 Full Program (logsumexp) | 0.00 | 21.43 | 46.98 | 65.96 | 52.02 | 49.20 | 6.03 |
| P7 Human Gold Program | — | — | — | — | — | — | — |

P7 is unavailable because no independent human review was performed; filling it from the rule parser would be circular.

### Forced-signal diagnostic (`lambda=1`)

| Method | R@1/20 | R@4/20 | R@8/20 | R@1/40 | Top1 Dist (m) |
|---|---:|---:|---:|---:|---:|
| P1 Flat | 21.84 | 47.09 | 65.33 | 51.58 | 49.40 |
| P2 Structured | 19.61 | 44.53 | 63.63 | 46.87 | 55.12 |
| P3 + Axis | 18.95 | 44.01 | 63.40 | 46.76 | 55.42 |
| P4 + Ordinal | 18.32 | 43.38 | 63.07 | 45.42 | 57.57 |
| P5 Object-aware | 13.35 | 36.04 | 57.84 | 36.86 | 65.71 |
| P6 Full | 17.43 | 43.75 | 65.78 | 42.31 | 61.24 |

This rules out the interpretation that lambda selection merely hid a useful structured signal.

## Relation-level result

The table reports frozen val_unseen sample count and P6 fixed-lambda change from P0.

| Relation | N | delta R@1/20 | delta R@4/20 |
|---|---:|---:|---:|
| left | 292 | -6.16 | -7.53 |
| right | 328 | -3.35 | +1.22 |
| front | 543 | -4.60 | -2.03 |
| behind | 458 | -5.46 | -7.42 |
| near/beside | 685 | -5.11 | -4.67 |
| between | 469 | -1.92 | -2.77 |
| ordinal | 206 | -7.77 | -0.97 |
| before | 8 | +12.50 | +12.50 |
| after | 9 | -11.11 | -11.11 |
| past | 5 | 0.00 | 0.00 |

The apparent `before` gain has only eight examples and is not evidence of a stable relation. Right-side R@4 is the only common-relation positive signal, but its R@1 worsens.

## Multi-relation result

At fixed lambda=1, Full Program R@1/20 was 17.75% for one relation, 15.98% for two relations, and 16.43% for three or more. Corresponding R@4/20 was 44.11%, 43.46%, and 45.00%. Multi-clause composition did not rescue the method; this is not CASE D.

## Axis, ordinal, and object representation

- Axis resolution harmed the fixed-weight result (P2 19.61% to P3 18.95% R@1/20). Start-to-anchor is deterministic but often not the linguistic frame intended by `left/right/front/behind`.
- Scoped ordinal execution further reduced R@1/20 to 18.32%. The failure is not proven to mean ordinal language is useless: the unresolved linguistic ordering axis and incomplete map object types remain confounded. However, the implemented scope/order mechanism adds no value.
- Object-aware execution was substantially worse than cell centers (13.35% R@1/20). Candidate regions often contain many compatible CityRefer objects, and max/logsumexp aggregation amplifies unrelated objects. This reproduces the broader warning that a B0 region cannot be reduced to an arbitrary mapped object.

## Failure taxonomy

Frozen val_unseen (2,697 episodes):

| Category | Share |
|---|---:|
| success within Top-4 after inference | 32.7% |
| correct parsed program but wrong reranking | 31.3% |
| B0 Top-16 candidate pool miss (>20 m) | 19.4% |
| no explicit/detected relation | 11.9% |
| ordinal scope/order failure | 3.6% |
| relation binding failure | 0.6% |
| anchor grounding failure | 0.4% |

These labels are deterministic diagnostic categories, not human adjudication. The largest actionable failure is executor/frame mismatch, not missing anchor lookup.

## Debug artifacts

Two hundred top-down figures were exported: 100 val_seen and 100 frozen val_unseen. Each includes instruction, anchor contour, start pose, selected axis, B0 candidates, and failure class. See `runs/spatial_program_eval_s0_20261004_r3/artifacts/debug_viz/` and `debug_records.json` for clause-level traces.

## Answers and decision

1. **Spatial Program vs flat:** no. Structured P2–P6 selected zero weight; forced use hurts.
2. **Clause binding as the bottleneck:** not established. Automatic binding failures are only 0.6%, while wrong reranking is 31.3%; a human Gold set is still needed to measure hidden binding errors.
3. **Reference axis:** materially affects ranking, but the current resolver affects it in the wrong direction.
4. **Ordinal:** scope/order repair did not help. The experiment cannot cleanly separate intrinsically weak ordinal language from incorrect latent axes without reviewed Gold programs.
5. **Object-aware vs cell center:** object-aware is worse.
6. **Achievable Gold upper bound:** unknown. It would be scientifically invalid to reuse parser output as Gold or inject GT target bearing/distance.
7. **Is explicit geometry sufficient?** Current evidence says no meaningful improvement over B0.
8. **Small LLM parser?** No. Gold-program headroom has not been demonstrated, and the deterministic executor signal is negative.
9. If Gold review later shows headroom, an LLM should be restricted to target/reference roles, relation-to-anchor binding, axis type, and ordinal scope JSON—never coordinates or candidate IDs.
10. Stop model scaling on this route until independent Gold review establishes a realizable upper bound.

**Decision: provisional CASE C, not CASE B.** Explicit executable geometry as currently represented does not improve B0, including multi-relation instructions. Because P7 is intentionally blocked pending human review, “provisional” is essential: the experiment rejects further parser/model complexity, but does not claim a measured Gold upper bound.
