# Global Landmark Prior for HETT

Branch base: `main`.

This branch isolates the landmark-grounding change from the heatmap and
trajectory-belief experiments.  The original HETT two-stage controller is kept
unchanged.

## What changes

The previous HETT map contained two tracking channels plus one landmark channel
constructed only from the landmarks resolved from the current instruction.

The new planning map has four channels:

```
0 current observation footprint
1 accumulated explored area
2 block-level global landmark prior
3 instruction-referenced landmark mask
```

The global prior contains all named geographic landmarks available in the
current CityNav block from the first navigation step.  The referenced mask is a
strict subset containing only the landmarks resolved from the current
instruction.

This mirrors the information separation used by SBFNav: geographic structure
is available globally, while instruction-specific landmarks are explicitly
marked rather than being conflated with the entire landmark map.

## Compatibility

`LandmarkNavMap.landmark_map` remains an alias for the referenced landmark map
so existing visualization code keeps working.

The map encoder now accepts four input channels.  Two ablation flags zero the
new channels without changing tensor shapes:

```
--disable_global_landmark_prior
--disable_referenced_landmark_mask
```

Referenced landmark centroids are also exposed by the environment as
`referenced_landmark_centroids`; the next commit injects these as explicit
landmark-anchor candidates while leaving the two-stage controller unchanged.


## Referenced landmark centroid anchors

The second commit injects the centroids of instruction-referenced landmarks as
explicit geographic anchor tokens.  This is intentionally implemented as
**context candidates**, not as a new controller or selector:

```
HETT tokens =
language
+ current visual feature
+ direction
+ global/referenced map feature
+ referenced landmark centroid anchors
+ original 7x7 grid candidates
```

The 7x7 grid candidates remain the only tokens decoded by the existing grid
classification head, and the original continuous-goal + Stage-1/Stage-2
execution path is untouched.  This isolates the effect of richer long-range
grounding information.

Centroid anchors are normalized global coordinates, padded to
`--max_referenced_landmarks 8`, and padding slots are masked in Transformer
attention.  Use

```
--disable_referenced_landmark_centroids
```

for the corresponding ablation.

This differs slightly from the full SBFNav proposal selector: SBFNav unions
referenced landmark centroids with field peaks as executable candidate goals.
Here they are first introduced only as Transformer geographic anchors so the
HETT controller can remain exactly unchanged.  A later heatmap-specific branch
can promote the same anchors into the executable Top-K proposal pool.
