# Heatmap Semantic-Geometric Candidate Selector

Branch: `2027-CVPR/heatmap-semantic-geometric-selector`

Base:
`experiment/heatmap-system-optimization-20e` via
`2027-CVPR/heatmap-candidate-visual-selector`.

## Goal

Keep the trained HETT heatmap as a high-recall proposal model, then train an
independent selector for the missing Top-K -> Top-1 step.

The selector intentionally has **no heatmap-score input**.  It receives only
candidate-specific evidence:

- accumulated RGB feature for the candidate when actually observed;
- full instruction tokens;
- candidate-to-agent explicit geometry;
- candidate-to-referenced-landmark explicit geometry;
- referenced-landmark name embeddings.

The referenced landmarks are exactly the same names and contours already used
by the heatmap referenced-landmark mask.  No target ID, target coordinate, GT
rank, or new annotation source is added to inference.

## Geometry

For every relative pair the selector uses the same five-dimensional encoding
as the SBFNav reproduction:

```
[dx, dy, distance, sin(beta), cos(beta)]
```

Coordinates are already normalized by the common HETT map extent, so this is
equivalent to computing metric geometry and dividing by map scale.

For each candidate, the token sequence is:

```
[SCR]
[candidate RGB]
[candidate <- agent geometry]
[instruction tokens...]
[landmark-1 name + landmark-1 -> candidate geometry]
...
[landmark-M name + landmark-M -> candidate geometry]
```

A 2-layer Transformer Encoder (default hidden=256, heads=8) produces the
candidate score from the SCR token.

Candidates are scored independently, so candidate order cannot encode the
answer.  Landmark tokens are also order-invariant when names and coordinates
are permuted together.

## Visual evidence

The branch preserves the previous selector's conservative visual memory:

- only heatmap hypotheses inside the current observed RGB footprint receive a
  candidate feature;
- observed candidate features are accumulated by 28x28 heatmap cell;
- unseen candidates receive no fabricated image feature;
- selector inference can abstain and preserve the raw heatmap Top-1.

## Training

Use the existing heatmap-20e checkpoint and freeze the base:

```bash
python multiagent/main.py \
  --mode train \
  --checkpoint /path/to/heatmap-20e/checkpoint \
  --candidate_selector \
  --candidate_selector_freeze_base \
  --epochs 23 \
  ...same heatmap arguments...
```

The selector loss remains:

```
L_selector = L_hard_top1 + 0.5 * L_listwise
```

Only evidenced candidates are supervised.  GT coordinates are used only to
construct the training target and offline diagnostics.

## Ablations

Two direct ablations are exposed:

```
--candidate_selector_no_geometry
--candidate_selector_no_landmark_text
```

Recommended comparison:

1. heatmap raw Top-1;
2. RGB + language selector, no geometry;
3. geometry + language with landmark text ablated as needed;
4. full RGB + language + landmark geometry selector.

The decisive metric is matched Top1@20 on exactly the same eligible candidate
sets, followed by rescue/regression and closed-loop SR/SPL only if the offline
selector gate passes.
