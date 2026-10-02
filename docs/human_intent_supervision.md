# Human-Intent Supervision Prototype

This branch is derived directly from
`2027-CVPR/hett-heatmap-two-stage`.

Its purpose is to separate useful **human navigation intent** from low-level
keyboard-control artifacts in CityFlight demonstrations before changing the
main heatmap controller.

## First-principles assumption

CityFlight human traces are collected from first-person flight with keyboard
control.  The raw path therefore mixes two kinds of signal:

- useful route and observation intent: where the pilot chooses to approach,
  large corrections, deliberate view/yaw changes, and arrival behaviour;
- interface artifacts: small zig-zag corrections, repeated nearly stationary
  points, reaction delay, and other key-press-level noise.

The raw demonstrations remain untouched.  This branch derives a compact intent
representation for supervision.

## Intent reconstruction

`multiagent/human_intent.py` performs three operations:

1. remove consecutive poses that are both spatially tiny and have negligible
   yaw change;
2. apply Ramer-Douglas-Peucker simplification to XY geometry;
3. force large yaw changes to remain as keyframes, even when the UAV barely
   moves.

This is intentionally not shortest-path replacement.  Large human detours and
view changes remain part of the intent path.

## Fixed physical horizons

Instead of splitting the remaining trajectory into five equal fractions, future
targets use stable metric horizons:

```
10 m / 25 m / 50 m / 100 m / final goal
```

A horizon is masked when less than that distance remains.  The final goal is
always valid.  Each target includes:

```
(x, y, sin(yaw), cos(yaw))
```

so first-person observation orientation can be supervised together with route
geometry.

## Why this is a separate branch

The heatmap execution logic is intentionally kept unchanged in the first
commit.  This lets us verify the reconstructed human targets independently
before coupling them to trajectory-belief execution.

## Auxiliary supervision integration

The second commit connects the reconstructed intent targets to HETT while
leaving heatmap Stage-1/Stage-2 execution unchanged.

A lightweight auxiliary head predicts one vector per fixed horizon:

```
[x, y, sin(yaw), cos(yaw)]
```

from the same fused feature used by HETT's goal predictor.  XY is supervised
with masked Smooth-L1 and viewing direction with cosine loss.  Invalid horizons
(longer than the remaining intent path) are masked; the final goal is always
valid.

Default loss weights:

```
intent XY  = 0.5
intent yaw = 0.1
```

The reconstructed intent path is cached per episode, so RDP/yaw-keyframe
extraction is not repeated at every rollout step.  Logs include
`intent_xy_loss`, `intent_yaw_loss`, and the projection distance from the
current student state to the human intent path.

This branch is deliberately an ablation-friendly intermediate step: if the
human-intent targets are useful, the same fixed-horizon representation can then
replace equal-fraction trajectory targets in the separate trajectory-belief
line.
