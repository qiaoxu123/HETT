# CityNav Scene Evidence Grounding Report

## Executive conclusion

This experiment is **CASE B**: map-scene and referenced-landmark Oracle evidence
has a large Static-B0 upper bound, but the current RGB evidence does not match
the correct instruction better than a controlled same-map wrong instruction.
The Phase-4 Gate therefore failed. Learned/contrastive matchers, actual B0
updates, multi-seed training, controllers, action prediction, and navigation
rollouts were intentionally not run.

The result is negative but diagnostic. A raw scene score rises as the teacher
trajectory approaches the goal, yet the rise is not instruction specific. The
same RGB often scores an equally plausible wrong instruction at least as high.
Geometry is already useful; current visual evidence adds noise to it.

## Scope and protocol

- Base: `dcfc90a89bbc2c7a87923ff2b6c4cc5aa85174f5`.
- Splits read: `train_seen`, `val_seen`, `val_unseen` only. `test_unseen` was
  neither loaded nor evaluated.
- RGB source: the existing HETT `cropclient` evaluated at the exact published
  teacher-trajectory position and direction. This is the project’s yaw-dependent
  top-down orthophoto observation, not a candidate-centred synthetic camera.
- Frames: at most one real pose from each `0–20`, `20–40`, `40–80`, and `>80m`
  distance bucket per episode.
- Regions: CityRefer polygons intersecting the current real FOV, top five by
  visible area. They localize content inside the current frame; the camera is
  never moved to a candidate or object.
- Visual encoder: the previous experiment’s partial-tuned
  `google/siglip2-base-patch16-256` checkpoint. No visual training was performed
  in this experiment.
- Hard negative: another instruction from the same map, chosen to have similar
  parsed requirements. Correct and wrong instructions are evaluated on the
  same RGB, so altitude, brightness, map, camera, and distance are controlled.

No CityFlight/AirSim executable or released trajectory RGB frames were present.
Therefore genuine oblique/FPV views were unavailable. No rotated orthophoto or
perspective warp was presented as FPV.

## Phase 1 — Scene Template census

The rule parser keeps target attributes, named anchors, context, geometry, and
ordinal terms separate. Statistics cover all 27,045 development instructions:

| Template field | Count | Coverage |
|---|---:|---:|
| Target visual attributes | 24,979 | 92.4% |
| Named anchor | 26,984 | 99.8% |
| Context object | 23,040 | 85.2% |
| Geometric relation | 23,802 | 88.0% |
| Ordinal | 4,344 | 16.1% |

Counts by split are stored in Phase 1 `scene_template_stats.json`. Parser
coverage is not evidence that the terms are visually recognizable.

## Phase 2 — Real-pose dataset and leakage audit

| Split | Episodes | Frames | 0–20m | 20–40m | 40–80m | >80m |
|---|---:|---:|---:|---:|---:|---:|
| train_seen | 600 | 2,256 | 569 | 556 | 570 | 561 |
| val_seen | 300 | 1,112 | 279 | 272 | 282 | 279 |
| val_unseen | 300 | 1,111 | 281 | 271 | 280 | 279 |

Episode, target-object, and generated-sample overlap are all zero between
splits. All 4,479 records have
`observation_source=teacher_trajectory_pose`; the candidate-centred audit is
false. The dataset contains 22,237 in-FOV region proposals.

## Phase 3 — Scene Evidence extraction

Whole-frame, region-max, and whole+region evidence were extracted for 26,716
images in 49.9 seconds. Peak allocated GPU memory was 1.62 GiB. This phase was
inference only.

## Phase 4 — Distance curve and first Gate

### Raw distance trend

Combined explicit matching becomes less negative near the goal:

| Split | >80m | 40–80m | 20–40m | 0–20m | Episode Spearman |
|---|---:|---:|---:|---:|---:|
| val_seen | -1.303 | -1.050 | -0.868 | -0.785 | -0.411 |
| val_unseen | -1.098 | -0.973 | -0.879 | -0.785 | -0.235 |

This trend alone is not grounding. Altitude and distance are strongly
correlated (`rho=0.660` seen, `0.637` unseen), while score also correlates with
altitude (`rho=-0.377` seen, `-0.164` unseen). The same-frame instruction
control is therefore decisive.

### Instruction-specific hard negatives

| Representation | Seen accuracy | Seen margin | Unseen accuracy | Unseen margin |
|---|---:|---:|---:|---:|
| Whole RGB | 48.2% | -0.035 | 43.1% | -0.077 |
| Region-aware | 49.5% | -0.016 | 45.1% | -0.036 |
| Whole + regions | 50.0% | -0.016 | 45.5% | -0.034 |

Region localization is slightly less bad than whole RGB, but none is above a
usable instruction-controlled baseline. The Gate required greater than 55%
same-map accuracy and failed.

### Which predicted attributes localize?

Combined representation, val_unseen:

| Evidence | Comparable pairs | Hard-negative accuracy | Margin |
|---|---:|---:|---:|
| Color | 1,017 | 50.2% | +0.001 |
| Size | 138 | 50.0% | +0.005 |
| Shape | 23 | 54.3% | +0.284 |
| Roof | 413 | 50.0% | 0.000 |
| Coarse semantic | 1,111 | 50.3% | +0.014 |
| Road/context | 818 | 47.4% | -0.024 |

The shape subset is too small to support a conclusion. No well-supported
attribute contributes instruction-specific localization, despite some having
high object-crop classification F1 in the previous benchmark.

## Target, Anchor, and Geometry

| Method | Seen accuracy | Seen margin | Unseen accuracy | Unseen margin |
|---|---:|---:|---:|---:|
| Target visual only | 49.8% | -0.018 | 46.8% | -0.021 |
| Anchor/context visual only | 50.2% | -0.008 | 47.4% | -0.024 |
| Target + anchor visual | 50.0% | -0.016 | 45.5% | -0.034 |
| Current pose ↔ named-anchor geometry | **58.0%** | **+0.192** | **64.0%** | **+0.256** |
| Visual + geometry | 57.0% | +0.176 | 59.7% | +0.222 |

Geometry is more useful than the visual signal. Adding current visual evidence
to geometry degrades unseen accuracy by 4.2 points.

## History

Simple causal fusion does not rescue the signal:

| Visual history | Seen accuracy | Unseen accuracy |
|---|---:|---:|
| Current frame | 50.0% | 45.5% |
| Last 3 mean | 49.7% | 44.8% |
| Last 5/full sampled history | 49.6% | 45.2% |
| Confidence-weighted | 49.7% | 45.2% |

Because single-frame evidence is not instruction specific, history merely
averages noise. No history Transformer is justified.

## Phase 7 — Static B0 Oracle upper bounds

The Oracle re-ranks the existing B0 Top-16 pool. Weights are selected only on
val_seen and applied unchanged to val_unseen.

### val_unseen

| Evidence | R@1/20 | R@4/20 | R@8/20 | Top-1 distance |
|---|---:|---:|---:|---:|
| Static B0 | 21.43% | 46.98% | 65.96% | 49.20m |
| Perfect target attributes | 22.28% | 51.84% | 71.19% | 44.25m |
| Perfect local scene/context | 59.55% | 78.53% | 80.42% | 26.57m |
| Perfect referenced-landmark geometry | 69.74% | 80.24% | 80.53% | 19.56m |
| Perfect scene + geometry | **69.86%** | **80.35%** | **80.57%** | **19.68m** |

The Top-16 ceiling is R@16/20 = 80.57%. Scene+geometry nearly reaches it.
Target appearance alone has little discriminative power, while neighborhood
context and referenced-landmark geometry have large theoretical value.

## Static B0 actual update

Not executed by design. Phase 4 failed before the B0 integration Gate, so the
measured actual B0 delta is **not available**, not silently treated as a gain.
Static B0 remains unchanged. The branch records this as
`status=skipped`, `static_b0_modified=false`, and `actual_b0_delta=null`.

## Learned and contrastive matchers

Not trained. A small matcher could fit the strong altitude/progress shortcut
while leaving same-map instruction discrimination at chance. The supervised
entry point records a skipped metrics file when the Phase-4 Gate is false.
Thus no claim is made that explicit matching is better than a learned or
contrastive matcher; only the former was eligible to run under the agreed
protocol.

## Answers to the research questions

1. **Does current RGB contain instruction-matching scene evidence?** Not
   reliably under the current top-down observation and attribute head. Generic
   scene confidence changes with progress, but correct-vs-wrong instruction
   accuracy is at or below chance.
2. **Does score increase near the target?** Raw score does, but the controlled
   margin does not. This is not accepted as grounding.
3. **Target or anchor/context?** Neither visual branch works. Oracle context is
   much more valuable than Oracle target attributes.
4. **Which attributes localize?** None of color, size, roof, coarse semantic, or
   road/context has reliable controlled evidence. Shape has only 23 unseen
   pairs and is inconclusive.
5. **Whole or regions?** Regions reduce the failure magnitude, but combined
   remains below chance on unseen; there is no successful representation.
6. **Explicit, learned, or contrastive?** Explicit failed the Gate; learned and
   contrastive were intentionally not trained.
7. **Is geometry more important?** Yes: 64.0% unseen controlled accuracy versus
   45.5% visual; the Oracle gap is also much larger.
8. **Does history help?** No. Three- and five-frame means do not beat the current
   frame.
9. **Oracle B0 gain?** Scene+geometry raises unseen R@1/20 by **+48.43 points**
   and R@4/20 by **+33.37 points**, close to the Top-16 ceiling.
10. **Actual B0 R@1/R@4 gain?** Not evaluated because the prerequisite Gate
    failed; no actual improvement is claimed.
11. **Failure source?** Primarily perception/scene representation and viewpoint,
    not lack of theoretical language/map discriminability. Target attributes
    themselves are weak; context and geometry are discriminative.
12. **Active Perception?** Worth a focused next experiment only if genuine
    oblique/FPV rendering becomes available. The current data cannot validate
    it, and top-down history alone is not useful.

## Decision

**CASE B: Oracle has a large gain, while the actual matcher is poor.** The next
step should improve perception and camera view, especially genuine oblique/FPV
observations of context and anchors. Do not connect the current visual score to
Static B0 or the ET controller.

## Reproducibility artifacts

Runs are under `/mnt/windows-data/hett-scene-evidence/`:

- `phase1_template_s0_20261004/`
- `phase2_teacher_rgb_s0_20261004/`
- `phase3_explicit_evidence_s0_20261004/`
- `phase4_distance_gate_s0_20261004_r5/`
- `phase5_matcher_gate_s0_20261004_r2/`
- `phase6_hard_negatives_s0_20261004_r2/`
- `phase7_oracle_s0_20261004_r3/`
- `phase8_b0_gate_s0_20261004_r2/`

Each supervised run contains commands, provenance, source snapshot, status, and
phase metrics. Phase 9 history is stored in Phase 4 metrics. Phase 10 was not
run because there was no eligible learned model to replicate.
