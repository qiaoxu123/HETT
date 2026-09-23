# Implementation status

## 2026-09-23: paired visual validation

- Added `scripts/build_altitude_view_benchmark.py` for same-pose multi-height
  crops, exact clipped footprint masks and digital-zoom controls.
- Added strict common-visibility semantic annotations so field-of-view changes
  cannot be confused with recognition improvement.
- Added an isolated RSRefSeg2 environment, evaluation config, per-sample metric
  and single-GPU supervisor.  Official/model assets are external to the repo.
- Pilot protocol and first geometry result are recorded in
  `ALTITUDE_VIEW_PROTOCOL.md`.
- The formal 128-episode-per-split geometry result and semantic environment
  validation are recorded in `ALTITUDE_VIEW_RESULTS.md`.
- The active-altitude policy remains unwired to HETT rollout until the frozen
  semantic comparison passes.

## Present in this worktree

- Pure, per-episode active-altitude controller.
- Four-level discrete altitude and phase definitions.
- Multi-scale target-belief fusion.
- Privileged teacher target builder with soft human-height supervision.
- Independent temporally confirmed stop rule.
- Optional confidence-gated semantic-prior interface suitable for frozen
  RSRefSeg2 outputs.
- Bounded quantized observation LRU design.
- Auxiliary PyTorch heads and masked loss definitions.
- Unit-test specifications for the above components.

## Deliberately not done

- No changes to the released HETT rollout, environment or checkpoint format.
- No RSRefSeg2 model dependency or weights copied into this repository.
- No HETT training has been launched from this worktree.
- Frozen RSRefSeg2 semantic inference is pending external asset download and
  release of the shared GPU; only the CPU geometry/image pilot is complete.
- No claims about navigation improvement or runtime improvement.

The next implementation turn should begin with Gate 0 in
`ACTIVE_ALTITUDE_DESIGN.md`, then wire one component at a time behind a single
feature flag while preserving the baseline path byte-for-byte where practical.
