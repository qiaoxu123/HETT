# Heatmap Top-K Candidate Visual Selector

Branch: `2027-CVPR/heatmap-candidate-visual-selector`

Base: `experiment/heatmap-system-optimization-20e`

## Why this branch exists

The dense heatmap already has high Top-K coverage, but its Top-1 is often wrong.
A larger heatmap score, distance to the referenced landmark, or another
landmark-derived prior does not provide genuinely new evidence: those signals
mostly repeat the same coarse localization bias.

This branch therefore separates the task into two roles:

1. **Heatmap = proposal only.** It produces diverse NMS Top-K spatial
   hypotheses.
2. **Candidate selector = discrimination only.** It receives candidate-specific
   RGB evidence that has actually been observed plus instruction tokens, and
   decides whether visual evidence is strong enough to replace the heatmap
   Top-1.

The selector does **not** receive:

- heatmap logits/probabilities/scores;
- referenced-landmark masks or landmark distance;
- target coordinates, target IDs, or GT rank.

GT is used only for the selector training loss and offline diagnostics.

## Candidate-specific RGB evidence

The existing Darknet frame tensor is spatial: `512 x 7 x 7`. For every
heatmap Top-K candidate, its normalized map coordinate is projected into the
current top-down camera view using the current pose, yaw, and altitude above
ground. Bilinear sampling retrieves a local 512-D feature only when that
candidate lies inside the current RGB footprint.

World/map convention is handled explicitly:

- normalized x increases with world +x;
- normalized y increases south, opposite world +y;
- image top is agent-forward;
- image left is agent-left.

No RGB is read from an unobserved map location.

## Multi-view evidence memory

Observed candidate features are accumulated in a per-rollout memory keyed by
the 28x28 heatmap cell. A candidate's selector feature is the mean of all RGB
features actually observed for that hypothesis so far.

Thus the selector can use evidence seen in an earlier replan even when the
candidate is no longer inside the current frame. Unobserved candidates remain
masked and receive no fabricated/global orthophoto feature.

## RGB-language selector

For each evidenced candidate:

```
candidate RGB feature (512)
        |
visual projection
        |
candidate query --------> cross-attention over full instruction tokens
        |
candidate logit
```

Candidate queries are scored independently, so candidate order cannot carry
target information. The module has no input path for the heatmap field score.

## Abstention instead of forced Top-1

The selector is allowed to override the raw heatmap Top-1 only when:

- at least `candidate_selector_min_visible` hypotheses have accumulated visual
  evidence (default 2);
- selector confidence >= 0.55;
- Top1-Top2 probability margin >= 0.10.

Otherwise the original heatmap Top-1 is preserved.

Most importantly, selector decisions are **never applied to student training
rollouts**. Training trajectories remain those of the heatmap baseline.
Selector reranking is applied only to student inference/evaluation
(`train_ml is None`).

## Training objective

Only currently evidenced candidates participate. The target is the nearest
**visible/evidenced** candidate to GT, and the sample is supervised only when
that candidate is within 20 m by default.

The loss is:

```
L_selector = L_hard_top1 + 0.5 * L_listwise
```

This directly trains the missing Top-K -> Top-1 step instead of adding another
smooth spatial heatmap loss.

## Recommended use with the existing heatmap-20e checkpoint

Do not restart the full 20-epoch HETT training.

Load the existing heatmap-20e checkpoint, enable the selector, and freeze the
base model:

```bash
python multiagent/main.py \
  --mode train \
  --checkpoint /path/to/heatmap-20e/checkpoint \
  --candidate_selector \
  --candidate_selector_freeze_base \
  --epochs 23 \
  ...same arguments used by the heatmap-20e run...
```

If the checkpoint reports completed epoch 20, `--epochs 23` adds three
selector-only epochs. Do **not** use `--resume_optimizer`: the selector has a
new optimizer state.

The freeze mode keeps BERT, Darknet, heatmap, HETT controller, and their
BatchNorm/dropout state fixed. Only `candidate_visual_selector.*` is updated.

## What to look at first

The logs report matched diagnostics on exactly the steps where the candidate
pool contains a good evidenced hypothesis:

- `visual_top1@20`
- `raw_top1@20`
- `visual_dist`
- `raw_dist`

This is the first gate. The selector is useful only if visual Top-1 clearly
beats raw heatmap Top-1 on the same candidate sets.

Closed-loop logs additionally report:

- selector trigger count;
- changed candidate count;
- rescue count;
- regression count;
- net rescue = rescue - regression.

## Recommended gate

Keep this direction only if all of the following hold on Val-Seen after tuning:

1. `visual_top1@20 > raw_top1@20` by a meaningful margin;
2. visual mean distance is lower;
3. rescue > regression;
4. closed-loop SR/SPL do not decrease;
5. the gain survives one frozen Val-Unseen evaluation.

If the matched visual selector cannot beat raw heatmap Top-1, do not add more
relation modules. That would indicate that the current RGB features themselves
do not contain enough target-discriminative information.

## Current scope

This branch intentionally stops before object-instance extraction or a new
active-view policy. It tests the more fundamental question first:

> once a heatmap candidate is actually observed, can candidate-specific RGB +
> language rank it better than the landmark-biased heatmap itself?

If this gate passes, the next step is to replace spatial cells with explicit
Building/Car object instances and let low-confidence states choose an
information-gathering observation waypoint rather than simply preserving the
heatmap Top-1.
