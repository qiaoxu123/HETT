# Global Landmark Prior in develop

The develop branch keeps the original HETT map pathway baseline-compatible and
adds block-level global landmarks as a zero-gated residual.

## Baseline-preserving map fusion

The original HETT input remains exactly:

```
tracking_current
tracking_explored
instruction_referenced_landmarks
        |
   original 3-channel MapEncoder
        |
      F_base
```

The global landmark occupancy prior is encoded separately:

```
all block landmarks
        |
  1-channel Global MapEncoder
        |
      F_global
```

Fusion is:

```
F_map = F_base + tanh(alpha) * F_global
```

with `alpha = 0` at initialization. Therefore a from-scratch model starts from
the original HETT map behavior instead of changing the first convolution from
3 to 4 input channels.

Use `--disable_global_landmark_prior` for the clean ablation.

## Referenced landmarks

Instruction-referenced landmarks remain the third baseline map channel and keep
the original HETT name-matching behavior so the corrected baseline is
comparable to the previous heatmap branch.

When the optional centroid-token ablation is enabled, centroids use the
annotated CityRefer object center instead of the arithmetic mean of contour
vertices.

Explicit centroid tokens are **off by default** because coordinates alone do
not carry landmark identity. They can be enabled only for ablation with:

```
--enable_referenced_landmark_centroids
```

## Formal heatmap resolution

Formal training scripts now default to `grid_size=7`. The previous inherited
5x5 script setting made one coarse cell about 82 m wide on the 410 m map and was
inconsistent with the heatmap/two-stage design.
