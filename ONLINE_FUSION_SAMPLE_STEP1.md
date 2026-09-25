# Online semantic-RGB fusion sample — Todo 1

## Result

The causal sample-construction gate passed on a small development pilot:

| Split | Episodes | Online RGB frames | Frames per episode |
|---|---:|---:|---:|
| train_seen | 8 | 40 | 2–8 |
| val_seen | 8 | 42 | 2–9 |

All 82 images are readable and have unique hashes.  Every episode changes RGB
after movement, and every frame has a non-empty pose-conditioned map frustum.
There are no sequence, remaining-budget, one-action pose-jump or recurrent
belief-reference errors.

## Causal sample contract

Each episode stores language, independent named landmarks and relations,
ranked predicted candidates, a target-free initial belief, and a sequence of
current RGB observations.  Every frame stores pose, remaining official action
budget, map-frustum visibility and a reference to the previous belief:

- step 0 reads the initial candidate belief;
- step `t > 0` reads fusion output from step `t - 1`.

The builder does not fabricate a target-supervised previous belief.  Runtime
training/inference must unroll the fusion model causally.

## Visibility limitation

The pilot has geometric in-frustum visibility but no verified depth/occlusion
label.  Therefore every negative-evidence mask is empty.  An in-frame visual
non-detection remains unknown rather than being interpreted as target absence.
Point-cloud or predicted-depth supervision can enable negative evidence only
after a later, separate visibility check.

## Anti-leakage

- Route destination: predicted rank-1 candidate only.
- Target coordinate/object identity: absent from every inference sample.
- Previous belief: initial prediction or preceding model output only.
- Splits read: `train_seen` and `val_seen` only.
- `val_unseen` and `test_unseen`: untouched.

Machine-readable result:
`runs/online_fusion_pilot_step1/summary.json`.
