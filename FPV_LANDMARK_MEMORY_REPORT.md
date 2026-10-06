# FPV / Oblique Landmark Appearance Memory — diagnostic gate report

## Outcome

**No valid FPV-vs-top-down or appearance-memory result is claimed.** The
released data has no cached human FPV trajectory frames. A local AirSim scene
was available, but its direct CityNav-pose rendering did not align with the
CityRefer map: on a low-altitude train_seen diagnostic pose the RGB was nearly
uniform, its segmentation image had one color, and the depth image was
constant 1.0. An exploratory translation changed the rendered scene, proving
that the raw pose transform is wrong; that translation was not validated or
used for evaluation. The full render was stopped at this data-quality gate.

This is an **inconclusive renderer/data-alignment result**, not evidence that
FPV, multi-view, or memory cannot help. Do not integrate an appearance-memory
module into HETT based on this run.

## Data and split audit

| Split | Existing teacher-pose top-down rows | FPV source | Formal FPV eval |
|---|---:|---|---|
| train_seen | 1,940 | No cached human FPV; AirSim re-render attempted | Not run |
| val_seen | 832 | No cached human FPV; AirSim re-render attempted | Not run |
| val_unseen | 891 | No cached human FPV; AirSim re-render attempted | Not run |

Only those three splits were opened. `test_unseen` was not read. The existing
top-down rows are from the RSRefSeg2 diagnostic's teacher-pose orthophoto
dataset. It is visibility-filtered and is therefore not a full-episode sample.
No model was trained or selected on `val_unseen`; no navigation/controller
code was touched.

The AirSim builder began processing the source manifest but was intentionally
interrupted after 1,400 observations / 7,631 unique FPV frames, before a formal
manifest or evaluation was accepted. These partial files are retained outside
the repository at `/mnt/windows-data/hett-fpv-landmark-memory/formal_v1/` for
debugging only; they are not an evaluation dataset.

## Renderer gate findings

- The local binary was `ENVs/env_1/env_1/LinuxNoEditor/AirVLN.sh`; it exposed
  `Drone_1/front_0` on AirSim port 30001.
- Scene code for `birmingham_block_1` was `b1`. The tested pose was a real
  train_seen Pose5D sample at about 14 m above map ground.
- Direct world mapping used by the inherited prototype (`x, -y, -z`, yaw
  sign-flipped) produced a near-white frame (RGB mean 248.5, standard
  deviation 5.6), one unique segmentation color, and DepthPerspective min=max=1.
- A one-sample exploratory XY translation `(x-350, -(y-500))` produced a
  textured scene (26 segmentation colors; RGB standard deviation 45.2).
  This is only a diagnostic coordinate search, not a calibrated transform; it
  must not be applied to validation or unseen data as a hand-tuned fix.
- The local scene listed 856 actors, but no validated actor-to-CityRefer
  registration was established. Consequently projected footprint masks do
  not provide trustworthy true visibility or occlusion labels.

See [renderer alignment smoke panel](artifacts/fpv_landmark_memory/renderer_alignment_smoke.jpg).
The panel explicitly marks the exploratory translated view as unvalidated.

## Smoke-only retrieval checks

The six-row smoke dataset (two rows per split) confirmed that dataset loading,
candidate-crop extraction, SigLIP2 scoring and metric serialization execute.
It is far too small and biased by manifest order to compare views. In the
SigLIP2 smoke, the two `val_unseen` rows had Top-1 0/2 on top-down and 0/2 on
FPV, with one or fewer visible projected positive crops; this cannot estimate
generalization. A separate three-row RSRefSeg2-frozen smoke had no
`val_unseen` rows. RSRefSeg2 Prompter-only was not run in this gate.

| Check | Samples | Result | Interpretation |
|---|---:|---:|---|
| SigLIP2 smoke, top-down | 6 total; 2/split | serialization succeeded | pipeline only |
| SigLIP2 smoke, FPV / oblique | 6 total; 2/split | serialization succeeded | pipeline only; invalid visibility geometry |
| RSRefSeg2 Frozen smoke | 3 total | serialization succeeded | no unseen coverage |
| RSRefSeg2 Prompter-only | 0 | not run | gate failed before formal model comparison |
| FPV memory / C1-C3 analysis | 6-row smoke only | no inference-grade result | history masks inherit invalid renderer registration |

For historical context only, the preceding top-down GT-crop benchmark reported
val_unseen Top-1 27.05% for SigLIP2, 47.36% for RSRefSeg2 Frozen, and 42.65%
for RSRefSeg2 Prompter-only. Those values are not paired comparisons with the
new AirSim frames and are not merged into the smoke metrics.

Full smoke artifacts are under `/mnt/windows-data/hett-fpv-landmark-memory/smoke/`.
The committed `metrics.json` and `metrics.csv` clearly label the gate as
inconclusive and do not present smoke figures as formal validation.

## Answers to the research questions

1. **Does FPV solve appearance ambiguity?** Unknown. Authentic CityFlight FPV
   is absent, and the local renderer pose-to-map registration is unresolved.
2. **Does multi-view help?** Unknown; no valid same-landmark multi-view set was
   accepted.
3. **Does appearance memory help C2 (currently invisible, previously seen)?**
   Unknown; the projected visibility/occlusion labels are not reliable.
4. **Should it be integrated into HETT?** No—not until camera/map alignment and
   depth/visibility are calibrated.

## Required next step

Obtain the original CityFlight/CityNav perspective render cache or the matching
Unreal scene-to-map georeferencing/calibration (per map, including camera
extrinsics and a working depth/segmentation modality). Then rerun a small
manual overlay audit before generating full train/validation frames. Only after
that gate passes should the planned Top-down/FPV/oblique and C1/C2/C3 memory
tables be computed.

## Provenance and compute

- Branch base: `2027-CVPR/rsrefseg2-grounding-diagnostic@124240e50d72e8c5a706be06d916e1a854158bfc`.
- No weights were trained in this experiment; trainable parameters: 0.
- SigLIP2 smoke runtime / peak allocated GPU memory: 8.1 s / 0.89 GiB.
- RSRefSeg2 Frozen smoke runtime / peak allocated GPU memory: 18.0 s / 8.06 GiB.
- AirSim renderer startup and smoke used about 4 GiB GPU memory. Full image
  generation and formal retrieval were not completed after the alignment gate
  failed.
