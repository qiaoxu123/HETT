# Gate 0: pure-logic validation

Date: 2026-09-23

## Result

Gate 0 passes at unit and component-integration level.  This validates the
standalone active-altitude building blocks, not their effect on HETT navigation
metrics.

Command:

```bash
/home/rental/20260922_1/Workspace/DATA/rsrefseg2/venv/bin/python \
  -m pytest -q tests
```

Result: `20 passed in 4.95s`.

Python byte-code compilation and `git diff --check` also completed without an
error.

## Covered behavior

- Controller starts at 120 m and changes at most one altitude level per step.
- Sustained high entropy can reopen a phase and ascend.
- Stop requires refinement, proximity, confidence, stability and temporal
  confirmation.
- Randomized controller rollouts preserve altitude and stop invariants.
- High-altitude evidence is broader and weaker than low-altitude evidence.
- Confidence-gated semantic logits are centered before late fusion.
- Human altitude remains a soft teacher rather than overriding distance-derived
  phase structure.
- Observation keys are quantized and the LRU remains bounded.
- Runtime composition connects semantic prior, belief update and control.
- Runtime state remains independent for different episodes in one batch.
- Auxiliary heads and masked losses form a finite, differentiable training
  path, including the all-inactive case.
- Paired-view geometry and paired semantic-analysis helper tests pass.

## Limits of this gate

- No legacy HETT rollout or environment code is changed or exercised.
- No checkpoint compatibility, navigation SR/SPL or runtime overhead is tested.
- Controller thresholds are still design values rather than validation-calibrated
  values.
- The learned altitude-action head is not yet connected to rollout decisions.
- RSRefSeg2 navigation integration remains disabled.

## Gate 1 entry condition

Preserve the fixed-50 m baseline path behind a feature flag, then compare it
against a scripted phase schedule and an oracle phase/altitude ceiling before
training a learned altitude policy.
