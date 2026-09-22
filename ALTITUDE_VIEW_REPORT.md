# Altitude-dependent observation visual audit

Interactive report: <https://hett-altitude-view-20260922.jlu-mcns-mec-6084.chatgpt.site>

## Scope

- Four real trajectories: two `val_seen`, two `val_unseen`.
- Five evenly spaced observations per trajectory.
- Compare fixed 50 m / direct 224, clipped human height / direct 224, clipped human height / 448 then area-downsampled to 224, and clipped human height / 896 then area-downsampled to 224.
- Human height is clipped to 20–120 m for this visual audit because the raw annotations include negative and extreme relative heights.

The network is not changed and no navigation metric is claimed by this audit. The page is intended to judge whether altitude-dependent field of view and supersampling preserve useful landmark and object context before running a training ablation.

## Reproduce assets

```bash
PYTHONPATH=. /home/tenant2/miniconda3/envs/AirVLN39/bin/python \
  scripts/build_altitude_view_samples.py --output /path/to/site/dist
```
