# Dense 28x28 Spatial Belief

This branch replaces the old 7x7 auxiliary heatmap path with a dedicated dense
spatial-belief branch while preserving the existing ET navigation path.

## Inputs

The belief branch receives four map channels, in this order:

1. current observation footprint;
2. cumulatively explored area;
3. all global landmark contours;
4. instruction-referenced landmark mask.

The existing ET map encoder still receives the legacy three-channel view
(current, explored, referenced), so the extra global-landmark channel does not
change the legacy controller representation.

## Belief branch

```
4-channel map
  -> compact spatial CNN
  -> adaptive 28x28 feature field
  -> token-level cross-attention with instruction
  -> 28x28 belief logits
  -> spatial softmax
  -> greedy NMS Top-K
```

The main 7x7 candidate/history tokens remain unchanged.

## Supervision

The dense belief uses a Gaussian target defined in metric coordinates rather
than grid-cell units. Defaults:

- field size: 28x28;
- Gaussian sigma: 20 m;
- NMS Top-K: 16;
- NMS kernel: 3.

The Stage-1 controller still executes the highest-ranked NMS hypothesis. The
remaining hypotheses are retained in trajectory diagnostics for later
candidate-selection experiments.

## Review boundaries

This change intentionally does not:

- add the SBF semantic-geometric selector;
- add RGB/height/trajectory channels to the belief CNN;
- alter Stage-1/Stage-2 switching;
- change the legacy 7x7 history aggregation;
- change the heatmap loss weight.

This keeps the experiment focused on dense spatial belief quality.
