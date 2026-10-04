# CityNav Visual Attribute Observability Benchmark

This experiment asks which instruction attributes are observable in UAV imagery
well enough to verify candidates proposed by Static B0. It does **not** train a
controller and does not learn RGB-to-goal coordinates.

## Protocol

The benchmark separates language requirements, visual evidence, and geometric
constraints. Objects and episodes are grouped before splitting; only
`train_seen`, `val_seen`, and `val_unseen` are accepted. `test_unseen` is rejected
by the dataset reader and census entry point.

The phases are gated:

1. instruction census, leakage audit, and hand-crafted features;
2. frozen DINOv2-small, existing HETT DarkNet, SigLIP, and SigLIP2;
3. partial tuning of at most the best two encoders;
4. view, height, and distance ablations;
5. whole-image, crop, and segmentation/isolation ablations;
6. multi-view fusion;
7. explicit attribute scoring of Static B0 candidates;
8. multi-seed verification of the selected configuration.

Each phase writes a standalone `metrics.json` under its supervised run. Runs
also preserve the source snapshot, commands, git diff, status, and GPU telemetry.

## Labels and limitations

- Size and shape are deterministic functions of the CityRefer contour.
- Color, roof mentions, and some context labels are weak labels parsed from the
  descriptions. They require manual sampling before being treated as ground
  truth.
- The masked representation uses the known global contour. It is an oracle
  geometry-isolation upper bound, not a learned segmentation result.
- Locally available orthophotos support genuine top-down crops at several
  heights. Oblique and forward-facing images require the CityFlight/AirSim
  renderer; rotating or warping an orthophoto is not an acceptable substitute.
- Attribute frequency is not observability. A reliable attribute must perform
  well on `val_unseen`, remain consistent across available views/heights, and be
  calibrated.

## Commands

```bash
PY=/home/ubuntu5/miniconda3/envs/AirVLN/bin/python

$PY scripts/supervise_experiment.py \
  --run-dir /mnt/windows-data/hett-visual-attributes/phase1 \
  --python "$PY" --phase visual-attributes-phase1

$PY scripts/supervise_experiment.py \
  --run-dir /mnt/windows-data/hett-visual-attributes/phase2 \
  --python "$PY" --phase visual-attributes-frozen \
  --pythonpath /mnt/windows-data/hett-visual-attributes/pydeps \
  --dataset-root /path/to/phase1/artifacts/dataset
```

The isolated `PYTHONPATH` is used because SigLIP2 requires a newer Transformers
release than HETT's navigation environment. It does not replace the project
environment or modify its installed packages.

## Interpretation rule

Any downstream gain is described as candidate verification from observed
attributes. It is not evidence that a vision model has learned a novel landmark
identity, and geometry-derived attributes are not credited to the visual model.
