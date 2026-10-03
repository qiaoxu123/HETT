# Language-Conditioned Goal-Approach FPV Dataset

This is an independent prototype for determining whether a language-described
goal matches the current view, whether the agent should stop, the view-relative
target bearing, and the far/approaching/near/arrival phase. It does not modify
the HETT navigation policy.

## Provenance and renderer limitations

Samples are selected directly from the raw CityNav MTurk `Pose5D` traces, so
`x, y, z, yaw, pitch` are retained. They are never reconstructed from HETT's
pitch-free `teacher_trajectory`.

The released CityNav/HETT data contains trajectories, CityRefer annotations,
orthophoto RGB, and DSM rasters. It does **not** contain the original CityFlight
browser renderer or cached human FPV frames. Two renderer families are exposed:

- `airsim_front` / `airsim_slanted`: genuine AirSim perspective rendering. A
  CityNav-compatible Unreal server must already be listening on port 41451.
- `orthophoto_heightfield`: an explicitly labelled 2.5-D perspective proxy from
  the published orthophoto. It is suitable only for pipeline smoke tests and is
  never described as an original human view or an AirSim frame.

## Samples and labels

The builder inherits `train_seen`, `val_seen`, `val_unseen`, and `test_unseen`
without frame-level random splitting. It samples trajectory points nearest to
80, 60, 40, 30, 20, 15, 10, and 5 metres plus the final trace point, removes
near-duplicate anchors, and forms progressive four-frame clips at `t-6, t-3,
t-1, t`. Every record includes an episode ID, raw poses, target and referenced
landmark coordinates, render provenance, and five task labels.

Positive/base examples are complemented by near-goal non-arrivals, yaw-shifted
wrong views, and same-map/same-type nearby wrong-language negatives. Geometric
arrival and visually confirmed arrival remain separate labels.

## Baseline

The baseline freezes `google/siglip-base-patch16-224`, fuses frame embeddings
with a GRU, optionally adds the SigLIP text embedding and Pose5D embedding, and
trains equal-weight cross-entropy heads for match, arrival, distance, bearing,
and phase. Supported ablations are:

- `current_fpv_only`
- `current_fpv_language`
- `history_language`
- `history_language_pose`

## Commands

Build the requested smoke splits (run from the repository root):

```bash
python fpv_goal_approach/build_dataset.py --split train_seen --max_episodes 100 \
  --render_source orthophoto_heightfield --output runs/fpv_goal_approach_smoke
python fpv_goal_approach/build_dataset.py --split val_seen --max_episodes 50 \
  --render_source orthophoto_heightfield --output runs/fpv_goal_approach_smoke
python fpv_goal_approach/build_dataset.py --split val_unseen --max_episodes 50 \
  --render_source orthophoto_heightfield --output runs/fpv_goal_approach_smoke
python fpv_goal_approach/analyze_dataset.py --data runs/fpv_goal_approach_smoke
python fpv_goal_approach/visualize_samples.py --data runs/fpv_goal_approach_smoke
```

With a compatible CityNav AirSim server, replace the render source with
`airsim_front` or `airsim_slanted`.

Run tests and a two-epoch ablation:

```bash
pytest -q tests/test_fpv_goal_approach_dataset.py
python fpv_goal_approach/train.py --data runs/fpv_goal_approach_smoke \
  --output runs/fpv_goal_approach_smoke/training --variant history_language \
  --epochs 2 --batch_size 32
```

On the local shared environment, use the newer RSRefSeg2 Python and protobuf
compatibility mode for SigLIP:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /home/ubuntu5/Workspace/rsrefseg2-runtime/venv/bin/python \
  fpv_goal_approach/train.py ...
```

Full metadata/images use the same commands without `--max_episodes`. Do not use
`test_unseen` for tuning; `--metadata_only` is available when only the final
split manifest is wanted.

Evaluation JSON includes seen/unseen match and arrival AUROC/F1, arrival false
positive rate, distance/bearing confusion matrices, distance buckets, and
near-goal/wrong-view/wrong-language hard-negative metrics. Training additionally
writes a wrong-language shuffle test and, for history models, a frame-order
shuffle test.
