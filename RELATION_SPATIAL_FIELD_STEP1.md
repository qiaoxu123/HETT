# Relation spatial field — Step 1

## Scope

This step validates only two target-free spatial fields on `val_seen`:

- strict `left/right/above/below` relations bound to one named landmark;
- `near` relations directly bound to one named landmark.

The field scorer receives the parsed relation, landmark boundary and query
points.  The ground-truth target is read only after scoring to calculate
metrics.  Neither `val_unseen` nor `test_unseen` is read.

The directional frame is the fixed 270-degree transform selected by the prior
`train_seen` reference-frame audit.  No coordinate transform is selected or
tuned in this step.

## Results

| Check | Result | Gate | Status |
|---|---:|---:|---|
| Direction target score greater than flipped relation | 59.04% | >=60% | Fail |
| Correct-vs-flipped candidate Hit@20 gap | +5.85 pp | >=5 pp | Pass |
| `near` target in top field quartile | 98.58% | >=50% | Pass |

Directional candidate Hit@20 was 21.81%, versus 15.96% after flipping the
relation.  Thus the direction field contains useful signal, but it does not
meet the predeclared consistency gate.

The directional breakdown exposes the main problem:

| Relation | Count | Correct > flipped |
|---|---:|---:|
| left | 78 | 60.26% |
| right | 73 | 47.95% |
| above | 23 | 73.91% |
| below | 14 | 85.71% |

For 281 `near`-only examples, the target field percentile averaged 95.45 and
98.58% of targets were in the top quartile.  However, selecting the nearest of
the existing five candidates gave 25.98% Hit@20, below their original-order
29.18%.  Therefore `near` is a reliable search-region prior, not a sufficient
standalone candidate ranker.

## Decision

**Step 1 overall: failed.**  Do not proceed to soft/topological relations yet.
The next action, only after explicit instruction, should diagnose the
right-relation frame/parser ambiguity and decide whether directional relations
must be softened by confidence rather than used as hard masks.

Machine-readable output:
`runs/relation_spatial_field_step1/results.json`.
