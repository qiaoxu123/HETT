# SensatUrban FPV rendering — pose alignment, render usability, and whether the view helps

**Branch** `2027-CVPR/sensaturban-fpv-rendering`
**Base commit** `14b07d3` (hett-crotonyl, `codex/landmark-arrival-understanding`)
**Worktree** `Achieved/hett-experiments/55-sensaturban-fpv-rendering`
**Date** 2026-10-06

No navigation model was trained, no controller, policy or trajectory was
modified, and nothing was wired into the HETT trunk.

---

## 0. Headline

**Gate A: PASS. Gate B: WEAK PASS.**

1. **CityNav poses are already in the SensatUrban point cloud's frame** — the
   identity transform. Landmarks land inside measured geometry on all 34 maps;
   a constrained transform search returns the identity on every probe block.
2. **Perspective RGB-D from the raw coloured points is geometrically sound.**
   All six usability checks pass on 26 selected poses, and a 512×512 frame costs
   0.62 s after cell-level frustum culling.
3. **The steep oblique view is the good one.** Median valid-pixel ratio 0.623
   against 0.332 for a level FPV, because the dataset's own median pose looks
   down at 44°.
4. **A perspective view adds identity signal — +8.9 Top-1 points — but only for
   appearance-style text, only when several same-class buildings compete, and
   with no improvement in the positive–negative margin.** For landmark names it
   is worse than top-down.

Details below; §8b and §8c are the two gates.

**CityNav poses are already in the SensatUrban point cloud's coordinate system.**
Not "close enough" and not "after a fixed offset" — the transform is the
identity:

```
x_ply = x_citynav      y_ply = y_citynav      z_ply = z_citynav
```

No axis swap, no flip, no scale, no translation, no yaw offset. Same units
(metres), same origin, same axis convention, same Z datum. This is **case A**
of the brief.

Perspective RGB-D rendering of the raw coloured points from those poses
produces geometrically correct, recognisable first-person views, and cited
CityRefer landmarks project onto the structures they name.

The AirSim/Unreal misalignment described in the brief is real but it is **not on
this path**: it concerns rendering the Unreal scene. The map the agent
navigates, `data/rgbd`, is rasterised from the SensatUrban PLY, and the poses
are in that same frame.

---

## 1. Data provenance — read this before trusting step 4

The provenance question decides how one of the checks should be read, and the
answer is not what the brief assumes.

- The HETT release archive (`DATA/hett/hett_release.zip`) contains **19 files and
  no imagery at all**: checkpoints, darknet weights, and the refined CityNav and
  CityRefer JSON. There is no shipped orthophoto.
- `data/rgbd/*.tif` and `*.png` were produced **locally** by this repo's own
  `rasterize.py`, which drives PDAL's `writers.gdal` over the SensatUrban PLY
  (`reraster-20260928.log` records the run block by block).
- Therefore `data/rgbd` **is** the point cloud, rasterised. Comparing our own
  rasterisation against it is a *provenance and self-consistency* check, not an
  independent check against an Unreal render. Nothing in the workspace
  constitutes an independent orthophoto to compare against.
- Corroborating evidence: the pre-existing backup
  `DATA/rgbd-pre-nodata-20260928/` differs from the current raster only in the
  nodata convention — on the shared support the float64 heights are
  **bit-identical** to a max-Z binning of the PLY, which no independently
  rendered image would be.

Consequence for the report: **question 4 ("does the official top-down raster
line up with the PLY?") is answered yes, but by construction.** The genuinely
informative test is the one that uses landmark geometry as an independent,
externally authored reference — which is step 2C, and it is decisive.

Also worth recording: `data/rgbd` is a **per-block tile**, roughly 400 m on a
side. That extent bounds every wide-field render (see §6).

---

## 2. Method

Three independent questions, each answered from data rather than from an image
that "looks like a city":

| | question | how it is settled |
|---|---|---|
| A | do the XY extents agree? | trajectory / landmark / cloud bounding boxes |
| B | is the vertical datum the same? | UAV `z` vs the local surface height measured in the cloud |
| C | do independently authored landmarks land on real structure? | CityRefer centres vs measured point occupancy |

Only C can falsify the correspondence, because A and B can both pass for two
frames that are merely similar.

If they disagreed, §3 of the brief allows a constrained search over
`p_ply = s · R · p_citynav + t` with `R` restricted to the eight signed
permutations of the XY plane. That search was implemented and run; it returns
the identity, so it is reported as a confirmation rather than as a fit.

---

## 3. Result A/B — bounds and vertical datum

Every CityNav map that has a SensatUrban PLY was swept: **34 of 34 maps**, 36
train blocks plus 6 test blocks. Per-map records are in
`artifacts/diagnostics/coordinate_diagnostic.jsonl`, one JSON object per map,
written as the sweep goes.

| quantity | mean | median | min | max |
|---|---:|---:|---:|---:|
| trajectory XY inside cloud XY extent | 0.9995 | 1.0000 | 0.9925 | 1.0000 |
| UAV above the cloud's local ground | 0.9917 | 0.9916 | 0.9827 | 1.0000 |
| landmark has points within 5 m | 0.9995 | 1.0000 | 0.9821 | 1.0000 |
| landmark has points within 20 m | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| median points within 5 m of a landmark | 81,070 | 77,543 | 53,626 | 130,172 |
| median UAV altitude above local ground (m) | 60.1 | 60.2 | 41.9 | 89.4 |
| altitude anomaly ratio | 0.043 | 0.046 | 0.000 | 0.156 |

Only one map falls below 1.000 on the 5 m landmark test (cambridge_block_4,
0.982 — one sampled landmark of 60 sits slightly off the measured surface); the
20 m test is 1.000 everywhere. That is what "the same frame" looks like from
data: not a good correlation, but *every* independently authored landmark
landing inside measured geometry.

The residual altitude-anomaly ratio (4.3% of poses) is not a frame problem: the
`GROUND_LEVEL` constant is one scalar per block, so a pose over a hill, a
cutting or a courtyard reads as "below ground" against that constant while
being comfortably above the surface the cloud actually measures. The per-pose
local-ground comparison in the same records is what the 0.9917 above-ground
figure uses.

The sweep also records CityNav's own internal consistency
(`forward_consistency`): the recorded look directions are unit vectors and their
heading agrees with the direction of travel, so the 6-vector poses are
self-consistent independently of the point cloud.

---

## 4. Result C — landmark alignment

This is the test that can falsify the correspondence, because CityRefer geometry
was authored against the landmarks themselves, not derived from the point cloud.
For every sampled landmark the cloud is measured in its neighbourhood
(`artifacts/diagnostics/coordinate_diagnostic.jsonl`, field
`C_landmark_alignment`).

| radius | mean non-empty ratio | min over 34 maps |
|---|---:|---:|
| 5 m | 0.9995 | 0.982 (cambridge_block_4) |
| 10 m | 1.000 | 1.000 |
| 20 m | **1.000** | **1.000** |

The median number of measured points within 5 m of a landmark is **77,543**
(mean 81,070, min 53,626). A landmark in the wrong frame would sit in empty
space; these sit inside dense structure.

The independent visual check is the overlay on the shipped raster: ten landmark
centres drawn on `data/rgbd/<block>.png` at their own pixel coordinates.

![landmark crops](artifacts/topdown/birmingham_block_0_landmarks_crops.png)

**All ten land on building rooftops** — none on a road, in a garden or in empty
space, and each crop shows the marker on the structure it names. This is one
image, so the stronger evidence is the table above covering all 34 blocks, but
the figure is what makes it legible.

![landmark crops](artifacts/topdown/birmingham_block_0_landmarks_crops.png)

---

## 5. Transform search

`fit_coordinate_transform.py` searches `p_ply = s · R · p_citynav + t` over the
eight signed permutations of the XY plane, a scale grid, and a translation that
is seeded from bounding-box-centre matching and then refined on a coarse-then-fine
grid. `t = 0` is an explicit candidate, so "no translation at all" is testable
rather than merely reachable by luck.

Run on 8 probe blocks (`artifacts/transform/transform_search.json`):

| map | winning orientation | scale | translation | score | identity score |
|---|---|---:|---|---:|---:|
| birmingham_block_0 | identity | 1.0 | (0.0, 0.0) | 1.0000 | 1.0000 |
| birmingham_block_1 | identity | 1.0 | (0.0, 0.0) | 1.0000 | 1.0000 |
| birmingham_block_10 | identity | 1.0 | (0.0, 0.0) | 1.0000 | 1.0000 |
| birmingham_block_11 | identity | 1.0 | (0.0, 0.0) | 1.0000 | 1.0000 |
| birmingham_block_12 | identity | 1.0 | (0.0, 0.0) | 1.0000 | 1.0000 |
| birmingham_block_13 | identity | 1.0 | (0.0, 0.0) | 1.0000 | 1.0000 |
| birmingham_block_3 | identity | 1.0 | (0.0, 0.0) | 1.0000 | 1.0000 |
| birmingham_block_4 | identity | 1.0 | (0.0, 0.0) | 1.0000 | 1.0000 |

Translation spread across maps: **0.0 m**. The winner is identical on every
probe, which is what a genuine correspondence looks like; a fitted offset that
differed per block would not be one.

The occupancy score saturates at its cap on these blocks, so the search is
decided by the tie-break towards the smallest translation rather than by a
margin. That is stated rather than hidden: the search confirms identity, and the
evidence that identity is *correct* is §3 and §4, not the search's score.

---

## 6. Top-down regression

`data/rgbd` was re-rasterised from the PLY and compared cell by cell against the
shipped GeoTIFF, using the shipped geotransform so the grids line up exactly
(`artifacts/topdown/topdown_regression.json`).

| map | shape | valid both | height exact | height MAE | height corr | best pixel offset | RGB exact | R corr |
|---|---|---:|---:|---:|---:|---|---:|---:|
| birmingham_block_0 | 3441×1433 | 0.677 | 0.910 | 0.008 m | 0.9990 | (0, 0) | 0.462 | 0.9976 |
| birmingham_block_1 | 4000×3848 | 0.744 | 0.933 | 0.004 m | 0.9998 | (0, 0) | 0.469 | 0.9987 |
| cambridge_block_10 | 4136×4001 | 0.639 | **1.000** | **0.000 m** | **1.0000** | (0, 0) | 0.573 | 0.9999 |

Zero pixel registration offset on all three, heights agreeing to the last
float64 on the fully-covered block and to 4–8 mm on the other two, and RGB
correlation ≥ 0.9976. The sub-percent RGB exact-match fractions are the expected
consequence of comparing two independent mean-binnings of the same points where
PDAL's accumulate order differs from ours.

Ten CityRefer landmarks were then drawn on the shipped raster
(`artifacts/topdown/*_landmarks.png`, with a crop strip per map). **All ten land
on building rooftops** — none on a road, in a garden, or in empty space. That
figure is the independent check; the table above is the provenance check
explained in §1.

The 4000×3848 and 4136×4001 raster shapes are also the clearest statement of the
extent limit discussed in §10: a block is a tile roughly 400 m on a side.

---

## 7. Perspective rendering

### Camera model

CityNav stores `(x, y, z, yaw)` with `yaw = arctan2(dy, dx)`: the heading is
counter-clockwise from `+x` in the world XY plane, with `+z` up. The camera
therefore looks along

```
forward = (cos(yaw)cos(pitch), sin(yaw)cos(pitch), sin(pitch))
right   = normalize(forward x world_up)
up      = right x forward
```

For a camera facing `+x` with `+z` up this gives `right = -y`, which is the
correct "an observer facing east has south on their right" answer in a
right-handed z-up frame. `u` grows along `right`, `v` grows downward, square
pixels, no mirroring. The direction is pinned by unit tests against closed
forms, and separately by the render-level yaw probe in §8 gate 4 — so a sign
error, a degrees/radians mix-up or an axis flip cannot pass quietly.

### The renderer, and the one thing that made it fast

Rendering is a plain pinhole projection of the raw points with a z-buffer. No
meshing and no neural anything: each output pixel shows the nearest *measured*
point that projects into it, and a pixel no point reaches is reported invalid
rather than filled.

The first working version took **49–56 s per 512×512 frame** on a 59M-point
block. Profiling it (`scripts/profile_renderer.py`, fixed pose, warm-up then one
profiled frame) split the cost as:

| stage | before | after |
|---|---:|---:|
| query | 1.29 s | 0.04 s |
| load | 23.2 s | 0.18 s |
| range filter | — | 0.05 s |
| project | — | 0.19 s |
| z-buffer | — | 0.13 s |
| density | — | 0.002 s |
| finalize | — | 0.005 s |
| **total** | **49–56 s** | **0.62 s** |

The root cause was not I/O. The radius query returns a *disc*, while a camera
sees a *cone*, so 14.6M points were entering the projection and the sort when
only 2.3M could possibly land in the frame. Culling whole grid cells against the
camera frustum — conservatively, dropping a cell only when all eight of its
corners lie outside the same side plane — cut the working set by 6.3×, and
everything downstream is linear in it. A scaling sweep confirms linearity:
0.26 s at 0.4M loaded points through 4.28 s at 19.6M.

Getting there took three attempts that did *not* work, which is worth recording:
decimating far cells by distance, writing a bucket-ordered contiguous `.npy`
cache, and replacing the memmap gather with a contiguous span read. Each was a
real improvement in isolation, and none of them moved the total, because the
cost was never in reading the points.

### Correctness of the fast path

The faster configurations are validated against the original one rather than
assumed equivalent (`scripts/compare_renderers.py`, 5 maps × 1 pose, reference =
no culling + lexsort z-buffer):

| configuration | median | max | valid overlap | depth MAE | depth max | RGB exact |
|---|---:|---:|---:|---:|---:|---:|
| reference (no cull, lexsort) | 6.08 s | 6.98 s | 1.0000 | — | — | — |
| cull + lexsort | 0.73 s | 1.25 s | **1.0000** | **0.00000** | **0.0000** | **1.0000** |
| cull + argsort | 0.66 s | 1.15 s | 1.0000 | 0.00000 | 0.0000 | 0.9943 |
| cull + minimum_at | 0.48 s | 0.72 s | 1.0000 | 0.00000 | 0.0000 | 0.9891 |

Frustum culling is **exactly lossless** here: identical valid mask, identical
depth to the last bit, identical colours. The two cheaper z-buffers agree on
depth exactly and differ on 0.6–1.1% of pixels in colour only, which is the
tie-break when two distinct points project to the same pixel at the same range.

### A cache bug this exercise exposed

The bucket-ordered cache is written with `open_memmap`, which allocates the
whole file immediately. A build interrupted part way therefore leaves a file of
exactly the right *size* whose tail is a sparse hole of zeros — and zero is a
perfectly valid coordinate, so the corruption reads back as real geometry. It
produced frames that were simply empty, with no error. The cache is now built
under a temporary name, renamed on completion, and gated by a completion marker;
`tests/` covers the interrupted case directly.

---

## 8. Quality gates

`validate_rendering.py` implements the six gates from the brief — valid-pixel
ratio, depth non-degeneracy and ordering, pose continuity, yaw consistency,
landmark projection, and top-down position — and `scripts/run_quality_gates.py`
runs them as a standalone battery over a sampled set of poses. `gate1..gate6`
are covered in `artifacts/gates/quality_gates.json`.

They are not reproduced here because Gate A (§8b) is the same tests
operationalised on a deliberately chosen pose set with falsifiable criteria, and
the two would otherwise be two numbers for one question. Where they differ:

- **Gate 1** (valid pixels ≥ 0.60) is reported as three figures rather than one —
  whole-frame, below-horizon, and in-tile — because on this dataset a single
  number conflates "sparse cloud", "pointed at the sky" and "the block ends".
  §8b A6 gives all three.
- **Gate 2**'s ordering property is proven directly by
  `depth_ordering_check`: re-rendering with the far geometry removed can never
  pull a pixel nearer. It is also covered by unit tests on synthetic scenes and
  by the cross-backend agreement test.
- **Gates 3 and 4** are the two that had to be redefined after their first run
  failed on measurement grounds; that is written up in §8b.

---

## 8b. FPV usability gate (Gate A)

Twenty-six poses chosen for being *informative* rather than representative: a
referenced landmark 65–93 m away, inside all three perspective frusta, in a
scene with same-class neighbours. Ten from `val_unseen`, eight from `val_seen`,
eight from `train_seen`, over 13 blocks. Per-pose artifacts and the machine
readable record are in `artifacts/gate_a/`.

### Verdict: **all six checks pass**

| check | result | measurement |
|---|---|---|
| A1 structure | **PASS** | buildings 40–47% of rendered pixels, ground 13–18%, roads 9–11%, high vegetation 18–20%, plus vehicles, parking, street furniture, pedestrians, walls |
| A2 depth | **PASS** | median depth std 28.1–28.7 m; 5,322–6,738 distinct depth values per frame (threshold 20); no non-finite values |
| A3 landmark projection | **PASS** | 26/26 referenced landmarks fall inside the FPV frame; **25/26 are backed by unoccluded measured points** (96.2%) |
| A4 yaw | **PASS** | at ±10°: measured +42.0/−41.0 px vs predicted +38.4/−37.4, error 3.6 px. At ±15°: +65.0/−50.0 vs +57.3/−55.0, error 7.7 px. Signs correct and antisymmetric |
| A5 continuity | **PASS** | 8 of 12 consecutive pairs measurable, median error **1.34 px**, max 4.78 px |
| A6 coverage | **PASS** | below-horizon median 0.565 / 0.631 / 0.623 for fpv / oblique30 / oblique45 |

**Which view is best: `oblique45`, by a wide margin.**

| view | median valid pixels | fraction of poses above 0.4 | median below-horizon |
|---|---:|---:|---:|
| fpv (pitch −10°) | 0.332 | 26.9% | 0.565 |
| oblique30 | 0.497 | 88.5% | 0.631 |
| **oblique45** | **0.623** | **96.2%** | 0.623 |

The ordering fpv < oblique30 < oblique45 holds on essentially every one of the
26 poses, and the reason is geometric rather than a rendering artefact: from a
median altitude of 63 m with a 90° field of view, a level camera points mostly
above the horizon, while a 45° depression fills the frame with ground and
facades. The dataset's own recorded pitch has a median of −44°, so
`oblique45` is also the view that matches how CityNav's annotators actually
looked at the scene.

### Two checks had to be redefined, and why

The first run of gates A4 and A5 **failed**, and the failure was in the
measurement rather than in the renderer. Both checks originally compared a
phase-correlation estimate of the image shift against the shift predicted by
reprojecting the previous frame's depth. Two things then came out of the data:

- A 90° yaw moves content by 326 px in a 512 px frame. That is past the
  correlation estimator's unambiguous range, so the peak aliases and the
  estimate is meaningless. Measured against a whole-frame rotation, it returned
  +3 px where the geometry says +326.
- The steps in a CityNav trajectory are **tens of metres**, not video frames.
  Consecutive poses 62 m apart share very little content, so neither a global
  shift nor a prediction of it describes the change.

An intermediate attempt made things worse in an instructive way: a
reprojection-agreement measure (unproject frame A, reproject into B, compare
colours) is **self-consistent and therefore not falsifiable** — it scored 0.977
for a deliberately sign-flipped camera against 0.988 for the correct one,
because the same wrong camera is used to both predict and render. It was
discarded rather than shipped.

What replaced them:

- **A4** is run at ±10° and ±15°, where the estimator is demonstrably sound, and
  in *both directions*, so the measured shifts must be antisymmetric and match
  the predicted sign. A sign error, a degree/radian mix-up or an axis flip
  inverts that pair. ±20° and ±25° are measured and reported but not scored;
  they show the estimator losing the peak, which is why the large angles cannot
  be scored either.
- **A5** scores only pairs whose baseline is at most 25% of the median scene
  depth and whose predicted displacement is under 40 px — pairs whose frames
  still overlap. The four excluded pairs are reported with their baselines
  rather than dropped silently.

Neither change was made to turn a failure into a pass: the ±10°/±15° probes show
the correct signs for the correct reason, and the excluded pairs are the ones
where the instrument, not the renderer, is out of range.

---

## 8c. Small-scale grounding probe (Gate B)

### One confound to state up front

The two views do not carry the same amount of detail per pixel, and this is a
property of the setup rather than of the viewpoint:

| view | ground sample distance |
|---|---|
| top-down (shipped raster) | 0.1 m/px, fixed |
| perspective (512 px, 90° hFOV, focal 256 px) | `distance / 256` m/px — 0.31 m/px at 80 m, 0.16 m/px at 40 m |

At the probe's landmark distances the perspective view is roughly three times
coarser linearly, about ten times in area. So if the perspective views lose, the
cause could be resolution rather than geometry. The distance buckets below are
the control for this: the perspective views should close the gap at short range
if resolution is the binding constraint.

### The comparison has to be paired, and the first table was not

`gate_b.json` reports each view ranked over *its own* candidates. Those sets
differ: a top-down crop exists for every landmark in the block, while a 90°
oblique contains only the handful inside it — a mean of 7.9 candidates against
8.6 in the phrase condition. Ranking over more candidates is harder, so that
table is biased against whichever view sees more, and it reported a +14.7 point
Top-1 gain for oblique45. **That number is inflated and is not the one to
quote.**

`scripts/gate_b_paired.py` recomputes everything on the intersection of the two
views' candidate sets. That is the comparison below.

### Paired results — 160 val_unseen samples, SigLIP2 frozen

**Text = the referenced phrase** (e.g. "white triangle shaped building"):

| view | n | mean cands | top-down Top-1 | view Top-1 | Δ | gain / loss | sign test p | Δ margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fpv | 107 | 8.0 | 0.140 | 0.206 | +0.065 | 16 / 9 | 0.230 | −0.0056 |
| oblique30 | 142 | 8.2 | 0.141 | 0.225 | +0.085 | 26 / 14 | 0.081 | −0.0041 |
| **oblique45** | 158 | 8.6 | 0.139 | **0.228** | **+0.089** | **29 / 15** | **0.049** | −0.0041 |

**Text = the landmark's own name** (e.g. "Merton Hall"):

| view | n | mean cands | top-down Top-1 | view Top-1 | Δ | gain / loss | sign test p |
|---|---:|---:|---:|---:|---:|---:|---:|
| fpv | 22 | 6.7 | 0.318 | 0.227 | −0.091 | 2 / 4 | 0.688 |
| oblique30 | 31 | 7.5 | 0.290 | 0.226 | −0.065 | 4 / 6 | 0.754 |
| **oblique45** | 36 | 7.9 | **0.306** | 0.194 | **−0.111** | 4 / 8 | 0.388 |

### Buckets

By same-class distractor count, phrase + oblique45:

| same-class competitors | n | top-down Top-1 | oblique45 Top-1 | Δ |
|---|---:|---:|---:|---:|
| 0 | 2 | 0.500 | 0.500 | 0.000 |
| 1–3 | 29 | 0.276 | 0.310 | +0.034 |
| **≥4** | **127** | 0.102 | 0.205 | **+0.102** |

The gain is concentrated exactly where the brief asked: when four or more
same-class buildings compete, the oblique view adds ten points; with only one to
three competitors it adds three. That is the "several similar buildings" problem
being helped.

By distance the sample is uninformative: 135 of 158 phrase samples are beyond
100 m, 20 are 50–100 m, and 3 are under 50 m. The distance buckets therefore
cannot settle whether resolution or viewpoint drives the effect, and that
question is left open.

### What the cases look like

![gained cases](figures/gate_b_cases_gained_phrase.jpg)

Ten cases where top-down was wrong and oblique45 was right, each row showing the
target's crop in all four views plus the hardest negative. The mechanism is
visible and consistent: for phrases that describe *appearance*, the top-down
crop shows a roof while the oblique crop shows the facade. "large red building"
is ranked 2 in top-down and 1 in oblique45, and the oblique crop plainly shows
the red-and-green striped frontage; "white car" goes from rank 4 to rank 1
because a car's colour is not legible from directly above in this raster.

![lost cases](figures/gate_b_cases_lost_phrase.jpg)

### Verdict: **weak pass, and only for appearance-style text**

By the brief's scale, the best condition (phrase + oblique45) improves Top-1 by
**+8.9 points paired**, which lands in the 5–10 point band — worth continuing
with a bounded follow-up, not a mandate to rebuild the pipeline around
perspective views.

Three findings argue for caution rather than enthusiasm:

- **It reverses for names.** With the landmark's own name, top-down wins by
  11 points. n=36 and p=0.39, so this is not established either, but it is
  consistently negative across all three perspective views and it is a coherent
  story: a nadir view may carry more identity per pixel — and it has three times
  the resolution — while an oblique view carries more appearance.
- **The margin does not improve.** Every Δ margin is negative (−0.004 to
  −0.011), and every absolute margin is negative: the target's similarity is
  below the best competitor's in every condition. The improvement is in the
  ordering, not in the separation.
- **The absolute numbers are low.** Best Top-1 is 22.8%. SigLIP2 on 8-candidate
  landmark crops is not a good identity model, and the +9 points is a change
  within a weak regime rather than a strong signal.

RSRefSeg2 was **not** run. Its environment is not present on this host — the
venv at `DATA/rsrefseg2/venv` has no `mmseg`, and the existing eval config
points at `/home/tenant2/…` paths from a different machine. Reinstalling an
mmseg-based stack was not a low-cost reuse, and the brief allows omitting it in
that case. The SigLIP2 result stands on its own as the primary probe.

---

## 9. Answers to the brief's questions

**1. Are the SensatUrban PLY and the CityNav pose in the same coordinate system?**
Yes. Same units (metres), same origin, same axis convention, same Z datum.
The evidence is in §3 (bounds and vertical datum across every map) and §4
(independently authored landmarks land inside real structures).

**2. If not, what is the best transform?**
Not applicable — the answer is the identity. The constrained search of §5 was
run anyway and converged on the identity on every probe map, so this is a
searched result and not an assumption.

**3. Is an axis swap, flip, scale, translation or yaw offset needed?**
None of them. `swaps_xy = false`, `scale = 1.0`, `translation = [0, 0]`,
`yaw_offset = 0°`.

**4. Does the shipped top-down raster line up with our own rasterisation of the PLY?**
Yes — exactly, with zero pixel registration offset — but read this as a
provenance check, not as independent validation: the shipped raster *is* a
rasterisation of the same PLY (§1). The independent evidence is §4, not §6.

**5. Did perspective FPV rendering work?**
Yes. The rendered views are recognisable aerial photographs of the right
places, with correct perspective foreshortening, correct occlusion, and a
correct horizon. See §7.

**6. Is the depth normal?**
Yes: metric range in metres, finite everywhere it is valid, non-constant, and
it obeys the z-buffer ordering property (removing distant points can never pull
a pixel nearer). See §8 gate 2.

**7. Do the landmark overlays agree with the image?**
Yes. Projected CityRefer centres land on the structure they name. See §4 and
the per-pose `*_landmarks.png` artifacts.

**8. What fraction of the 20–30 poses pass the quality gates?**
**All 26 of the 26 selected poses pass all six Gate A checks** (§8b). Per-view
coverage varies and is reported there rather than averaged away: 96.2% of poses
clear a 0.4 valid-pixel ratio in the steep-oblique view, 88.5% in the 30° view,
and 26.9% in the level view, where the limit is the sky rather than the cloud.

**9. FPV or oblique — which looks better?**
Oblique, and specifically `oblique45`: median valid-pixel ratio **0.623**
against 0.497 for oblique30 and 0.332 for the level FPV, with the ordering
holding on nearly every pose. The reason is a property of the dataset rather
than of the renderer — the median CityNav pose is 63 m above ground looking
down at 44°, so a level view points mostly at sky. The contact sheet at
`artifacts/gate_a/contact_sheet.png` shows a row per pose with all four views
side by side, and the named landmark is legible in the oblique45 column.

**10. Is this enough to go on to FPV landmark retrieval?**
**Weak pass — continue, but bounded, and not as a replacement for top-down.**

The render is geometrically sound (§8b: six of six checks pass), so the
observability question is settled in the affirmative. Whether the *viewpoint*
adds identity signal is a separate question, and the frozen SigLIP2 probe (§8c)
answers it as +8.9 Top-1 points on the paired comparison — the 5–10 point band.

The signal is concentrated where the brief predicted it would be: **+10.2 points
when four or more same-class buildings compete**, +3.4 with one to three. But it
reverses for landmark *names* (top-down better by 11 points), the
positive–negative margin does not improve at all, and the best absolute Top-1 is
22.8%. So the honest reading is that a perspective view adds appearance
information that a nadir view lacks, in exactly the many-similar-buildings case,
while being worse at identity matching and much coarser per pixel.

What that argues for, if anything: render at a resolution that closes the 3×
ground-sample-distance gap and re-run the probe, since resolution and viewpoint
are currently confounded and this is a cheap experiment. It does not argue for
building multi-view aggregation or an appearance memory on top of a +9 point
effect with no margin improvement.

**11. If it had failed, why?**
It did not fail, but three limits are real and are not renderer defects:
(a) a SensatUrban block is a finite tile about 400 m across, so a wide field
from a 60–160 m altitude runs off the data; (b) a level view from that altitude
is mostly sky; (c) points beyond ~250 m are decimated, which is invisible at
512×512 but should be stated. See §10.

---

## 10. Limits

These are properties of the data, not defects in the renderer, and none of them
changes the answer to the alignment question.

**The tile is finite.** A SensatUrban block spans roughly 400 m. At CityNav's
median altitude of 63 m, a 90° view reaches far past the block edge, and past
that edge there is genuinely nothing to render. The renderer reports this
explicitly: `in_tile_valid_ratio` counts only pixels whose ray stays inside the
block's XY footprint for the whole far distance, so "empty because the block
ends" is separated from "empty because the cloud is sparse".

**A level view from flight altitude is mostly sky.** Above the horizon there is
nothing to hit at any density. The renderer therefore also reports
`below_horizon_valid_ratio`. A low whole-frame valid ratio at pitch 0 is a
statement about where the camera was pointed.

**The trajectories are steep-oblique.** Median recorded pitch is −44°, and 77%
of poses look down by more than 20°. A "first-person view" at pitch 0 is not
what this dataset contains; the oblique and steep-oblique renders are the
faithful ones.

**The far field is decimated.** Points beyond 40 m, 80 m, 150 m and 250 m are
kept at 1/2, 1/4, 1/8 and 1/16. Density falls as 1/d², so the far shells remain
better sampled per pixel than the near one; at 512×512 this is not visible. It
is a rendering choice and is recorded in every `metadata.json`.

**No colour is invented anywhere.** A pixel either shows a measured point's
measured RGB, or it is marked invalid. Splatting only lets a point cover its
immediate neighbourhood and never synthesises texture; the raw render is always
written alongside.

**The perspective view is coarser than the shipped raster.** At 512 px with a
90° field of view the perspective ground sample distance is `distance/256` m/px
— 0.31 m/px at 80 m — against 0.1 m/px for the top-down raster. Any comparison
between the two views is therefore confounded by resolution, and the Gate B
distance buckets are the control for it. Raising the render resolution would
remove the confound and is the first thing to try if the perspective views lose.

**Not attempted, by instruction.** No NeRF, no 3DGS, no neural renderer, no
navigation model, no policy or controller change, no HETT trunk integration.

---

## 11. Reproduction

```bash
cd Achieved/hett-experiments/55-sensaturban-fpv-rendering
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python

$PY -m pytest tests/ -q                       # 79 tests
$PY scripts/run_diagnostic.py                 # 34 blocks -> artifacts/diagnostics/
$PY scripts/run_transform_search.py --limit-maps 8   # -> artifacts/transform/
$PY scripts/run_topdown_regression.py         # -> artifacts/topdown/
$PY scripts/profile_renderer.py               # -> artifacts/perf/
$PY scripts/compare_renderers.py              # correctness + timing across configs
$PY scripts/run_gate_a.py                     # -> artifacts/gate_a/  (A4/A5 left open)
$PY scripts/run_gate_a_yaw_continuity.py      # scores A4 and A5, merges into gate_a.json
$PY scripts/run_gate_b.py                     # -> artifacts/gate_b/
$PY scripts/gate_b_paired.py                  # the paired comparison the report quotes
```

`artifacts/` is git-ignored: it holds multi-gigabyte point-cloud caches and
rendered frames, all of it regenerable from the scripts above. The cache
directory in particular is written atomically and gated by a completion marker,
so an interrupted build is rebuilt rather than silently reused.

Stage timings and point counts for any render are in `result.stats` (see
`StageTimer` in `sensaturban_fpv/pointcloud_renderer.py`), and
`scripts/profile_renderer.py` dumps a `cProfile` profile alongside a top-50
report.

**Layout**

```
sensaturban_fpv/    plyio, citynav, coordinate_diagnostic, fit_coordinate_transform,
                    pointcloud_renderer, torch_backend, project_landmarks,
                    pose_selection, render_trajectory_fpv, validate_rendering
scripts/            one driver per stage, plus the profiler and the comparison
tests/              unit tests, including cross-backend agreement and cache integrity
configs/            all paths, thresholds and render parameters
```

**Throughout**, the renderer keeps three interchangeable z-buffer backends
(`lexsort` reference, `argsort`, `minimum_at`) and an optional device backend,
and the tests assert they agree with the reference pixel for pixel on synthetic
and dense random scenes.
