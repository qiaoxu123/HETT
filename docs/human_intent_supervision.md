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
