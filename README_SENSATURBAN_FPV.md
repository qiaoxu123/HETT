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
