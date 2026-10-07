# 55 — SensatUrban + CityNav first-person RGB-D rendering

Can a CityNav `Pose4D` be mapped into the SensatUrban point cloud's coordinate
system, and does a perspective projection of the raw coloured points then give a
geometrically consistent first-person view that landmark grounding could be
built on?

No navigation model is trained here, no controller or trajectory is modified,
and nothing is wired into the HETT trunk. The poses are read, transformed, and
rendered.

## Answer in one line

Yes, and the transform is the identity: CityNav poses are already in the
SensatUrban PLY frame, so the earlier AirSim/Unreal misalignment does not apply
to the pose-to-point-cloud path at all. See
[`SENSATURBAN_FPV_RENDERING_REPORT.md`](SENSATURBAN_FPV_RENDERING_REPORT.md).

## Layout

```
sensaturban_fpv/
  plyio.py                    block access (memmap) + cached XY bucket index
  citynav.py                  episodes, poses, CityRefer landmarks
  coordinate_diagnostic.py    same frame?  bounds / z / landmark occupancy
  fit_coordinate_transform.py search p_ply = s R p_citynav + t over 8 orientations
  pointcloud_renderer.py      pinhole projection, z-buffer, splatting, top-down bins
  project_landmarks.py        landmark projection + visibility estimate
  render_trajectory_fpv.py    per-pose artifacts and metadata
  validate_rendering.py       the six quality gates
scripts/                      thin CLI drivers, one per stage
tests/                        unit tests, run with pytest
configs/sensaturban_fpv.json  all paths and thresholds
artifacts/                    every output, including the cached block indexes
```

## Running

```bash
cd hett-experiments/55-sensaturban-fpv-rendering
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python

$PY -m pytest tests/ -q                             # 79 tests
$PY scripts/run_diagnostic.py                       # 34 blocks: same frame?
$PY scripts/run_transform_search.py --limit-maps 8  # constrained transform search
$PY scripts/run_topdown_regression.py               # self-rasterised vs shipped raster
$PY scripts/profile_renderer.py                     # per-stage timing, scaling sweep
$PY scripts/compare_renderers.py                    # fast paths vs the reference renderer
$PY scripts/run_gate_a.py                           # is the render usable at all?
$PY scripts/run_gate_a_yaw_continuity.py            # scores A4/A5 and merges them in
$PY scripts/run_gate_b.py                           # does the view add identity signal?
$PY scripts/gate_b_paired.py                        # paired recompute on shared candidates
$PY scripts/gate_b_cases.py --variant phrase        # case galleries
```

`run_quality_gates.py`, `run_render_poses.py`, `run_splat_comparison.py` and
`run_vision_probe.py` are the earlier full-battery drivers for the six
rendering gates, the per-pose artifact set, the raw-vs-splat comparison and a
standalone SigLIP probe; `run_gate_a.py` and `run_gate_b.py` are the focused
equivalents and are what the report quotes.

`run_transform_search.py` writes `artifacts/transform/best_transform.json`, which
every later stage picks up automatically; without it they fall back to identity.

## Reusing the repo's own code

Landmarks are loaded through `gsamllavanav.cityreferobject.get_city_refer_objects`
so `position` / `dimension` / `contour` have one definition in the tree. The
`Pose4D` and yaw conventions implemented in `pointcloud_renderer.Camera` follow
`gsamllavanav.space`. The block-level orthophoto in `data/rgbd` was itself
produced by this repo's `rasterize.py`, which matters for how step 4 should be
read — see the report.

## Operational notes

- `artifacts/cache/*.grid*.npz` holds the per-block XY bucket index (built once,
  a few hundred MB per block). It lives here rather than beside the source PLYs
  because the dataset tree is shared.
- Rendering decimates points beyond 40 m and 80 m (`render.lod`); density falls
  as 1/d², so the far shells stay better sampled per pixel than the near one.
  Set `lod` to `null` to render every point.
- `render.splat_radius` is kept at 0 for the artifacts the gates judge, so the
  measured coverage is the raw coverage. Enhanced renders are written alongside
  the raw ones and never replace them.

## Pushing from this host

`/etc/gitconfig` rewrites `https://github.com/` to a `ghfast.top` proxy that
wants its own credentials, and there is no sudo to change it.  Push directly
instead, letting `gh` supply the token:

```bash
GIT_CONFIG_NOSYSTEM=1 git -c credential.helper='!gh auth git-credential' \
  push https://github.com/qiaoxu123/HETT.git HEAD
```

The branch was pushed to `qiaoxu123/HETT` because the original `origin`
(`qiaoxu123/2027-CVPR`) no longer exists.

## Resolution-controlled viewpoint test

`run_resolution_control.py` re-measures the Gate B paired samples with the
resolution confound controlled — the oblique view at 512/1024/1536/2048 px, and
the top-down crop degraded to 0.2 / 0.3 m/px and, per sample, to the ground
sample distance the oblique view has at that landmark's range. Every source uses
the same physical crop extent, so the landmark's share of the crop is equal by
construction.

```bash
$PY scripts/run_resolution_control.py                  # long; --shard i --shards n parallelises
$PY scripts/merge_resolution_shards.py                 # de-duplicates the shards
$PY scripts/analyze_resolution_control.py --variant phrase
$PY scripts/analyze_resolution_control.py --variant name
$PY scripts/resolution_cases.py --variant phrase
```

Note that the workers are CPU-bound and the runtime is dominated by the largest
blocks: a sample on a 100M-point block reads a ~1.3 GB span per render, four
times over.

## Dual-view fusion

`run_fusion_data.py` produces candidate-level SigLIP2 scores for both views at
several top-down ground sample distances across the three splits;
`analyze_fusion.py` chooses every operating point on `train_seen`/`val_seen` and
scores `val_unseen` once. The choice discipline is structural, not a convention:
the selection helpers are only ever handed the validation splits.

```bash
$PY scripts/run_fusion_data.py --shard 0 --shards 4   # parallelises; resume-safe
$PY scripts/analyze_fusion.py --variant phrase
$PY scripts/analyze_fusion.py --variant name
```

Note the `pgrep -f` trap when chaining these: a wrapper whose own command line
contains the script path matches itself and waits forever.

## 3D-anchored multi-view target entity grounding

The unit of this round is a target entity of *any* CityRefer type — a building, a
car, a wall, a parking area, a piece of street furniture.  Its 3D support is
measured from the point cloud, projected into the top-down raster and the oblique
frame at two scales, and turned into pooled components (tight appearance, context
appearance, surroundings, single-view-exclusive patches, measured geometry) plus
one learned component: the fusion of the two views' patch tokens whose pairing
came from the world coordinates.  The text is the query in a single attention
layer over those components, so nothing anywhere branches on an entity's type.

```bash
$PY -m pytest tests/ -q                              # 103 tests
$PY scripts/check_entity_projection.py --per-group 20 # is the anchor right for cars too?
$PY scripts/run_entity_grounding_data.py --shard 0 --shards 4   # parallelises, resume-safe
$PY scripts/train_entity_grounding.py --variant phrase
$PY scripts/analyze_entity_grounding.py --variant phrase
$PY scripts/entity_grounding_cases.py --variant phrase
```

Entity support comes from a recorded ladder: annotated box, intersected with the
annotated footprint and with the point cloud's own semantic class, then wider
boxes, then a radius around the annotated centre.  Which rung was used is
returned with every entity, because a car whose support is really a patch of road
must not be indistinguishable from one that was localised.

The label cache (`artifacts/cache/*.labels.npy`) is separate from the sorted
xyz/rgb cache and has its own marker, so adding it does not invalidate indexes
that already exist — rebuilding those means re-reading multi-gigabyte blocks.

### What the entity round found

`GeoAligned` (3D-anchored patch fusion) reaches Top-1 0.338 on `val_unseen`
against 0.320 for the best masked single view — +1.75 points, McNemar p = 0.81 —
and the permuted-pairing control scores 0.343, so the world-coordinate pairing
adds nothing measurable. See §8f of the report.

One implementation defect is worth knowing about if you extend this code: the
pooled components stored per sample are already SigLIP2 pooling-head *outputs*,
while the patch tokens are head *inputs*.  `make_model` keeps those two spaces
apart and applies the frozen head exactly once, to the patch fusion.  Applying it
to a pooled component as well puts the learned methods behind a random
projection of the features they are meant to beat, which produces a negative
result that has nothing to do with the data.  Two unit tests pin the component
mean at initialisation and the uniform attention.

### What the feature-sufficiency diagnostic found

See [`FEATURE_SUFFICIENCY_REPORT.md`](FEATURE_SUFFICIENCY_REPORT.md). The short
version: with the target region correctly masked, the colour and texture of that
region land at chance (0.104 against 0.10), the instruction parser's relation
flags land at chance (0.107), and no deployable combination beats the zero-shot
text cosine on the masked SigLIP feature (0.325). The anchor oracle — knowing
which entity the language refers to — reaches 0.609, and rescues 64% of the
Top-4-but-not-Top-1 samples. The bottleneck is anchor grounding, not appearance.

The diagnostic also caught a sampling artefact worth remembering: the candidate
list is built around the reference, so a relation block measured over *the
candidate list* scored 0.685 with 3.8k parameters and no scene information. Over
the whole map the same block scores 0.201.

```bash
$PY scripts/run_feature_sufficiency_data.py           # ~7 min
$PY scripts/train_feature_probes.py                   # 23 arms x 3 seeds
$PY scripts/train_feature_probes.py --hidden 0 --tag linear
```

### Language anchor grounding

See [`LANGUAGE_ANCHOR_GROUNDING_REPORT.md`](LANGUAGE_ANCHOR_GROUNDING_REPORT.md).
The parser splits instruction into target / anchors / relations / attributes
(86% yield an anchor; 1.96 relation words per instruction), and 36% of samples
name an entity in their own block. The deployable chain — ground the anchors,
score candidates by their relation to them, fuse with the masked visual score —
reaches 0.357 against the 0.320 baseline, **+3.7 points: FAIL** by the round's
own threshold, and the reason is specific: removing the relation word from the
reasoner's input *raises* the score to 0.370, while removing the geometry drops
it to exactly the baseline. The gain is a proximity prior, not relation
reasoning, and the oracle ladder inverts — handing the reasoner the true anchor
scores worse than the predicted one — which localises the failure to the
relation reasoner.

```bash
$PY scripts/run_anchor_grounding_data.py    # ~50 s
$PY scripts/train_anchor_grounding.py       # ~5 min
```

### Reference binding and relation semantics, v2

See
[`REFERENCE_BINDING_RELATION_V2_REPORT.md`](REFERENCE_BINDING_RELATION_V2_REPORT.md),
with [`REFERENCE_BINDING_AUDIT_V2.md`](REFERENCE_BINDING_AUDIT_V2.md) and
[`RELATION_SEMANTICS_AUDIT.md`](RELATION_SEMANTICS_AUDIT.md).

The previous round ended by charging its own reasoner with not implementing the
relation. This round measures the corpus instead of assuming it, and finds the
stronger statement: **there is no consistent convention to implement.** Fitting a
geometric direction per relation word reaches paired AUC 0.584 against 0.575 for
randomly relabelled relations — the same number, meaning a model told which word
was used cannot place the target better than one told nothing. The largest-n
relation in the corpus (`on`, n=697) sits at 0.522. The compass words occur 26,
10, 0 and 1 times.

The audit also removes a confound both this round and the last inherited: the
candidate list is stored as `[referenced] + distractors`, so the answer sat at
index 0 and any tie broke toward it. On the corrected order the previously
reported "proximity prior" does not survive either — every phrase's distance
separation lands between 0.46 and 0.56.

Two structural facts about the input are worth keeping. `object_ids` is the
**target**, not a reference set — one entry in all 27,045 records, always at a
`target_position` — so it cannot be used as an anchor set. And landmark names are
heavily shared: 41.6% of anchor phrases name more than one entity, the largest
single group being 15 road segments sharing one name, so binding here is
disambiguation rather than matching.

The gate fails and, per the brief, no visual fusion was run: a fusion number on
top of a reasoner that cannot rank the target given the right anchor would
measure the visual baseline it was added to.

```bash
$PY scripts/audit_reference_binding.py       # ~4 min
$PY scripts/run_relation_v2_data.py          # ~15 s
$PY scripts/audit_relation_semantics.py      # ~2 min
$PY scripts/train_relation_v2.py             # ~2 min
```
