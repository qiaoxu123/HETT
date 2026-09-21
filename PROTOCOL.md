# Explicit constraint-graph grounding protocol

## Question

Does zero-shot decomposition into a referenced landmark and a spatial relation transfer
better than a learned sentence/token selector across unseen city blocks?

## Method

Use the same CityRefer landmark names, contours, 9 direction fields, 6 contour-distance
fields, 64×64 grid, 20 m hit radius, and NMS as experiment 28. Parse each instruction
into one relation with two fixed parsers: a lexical parser and local Qwen2.5-VL-3B in
text-only greedy mode. Apply a deterministic geometry operator to the referenced
landmark fields. There is no training and no target coordinate, target contour,
validation label, RGB, or trajectory input.

Evaluate full val_seen and val_unseen only. Report overall and per-relation Top-1 Hit@20
and Top-5 Recall@20. This is a static first-stage localization test, not navigation SR.
