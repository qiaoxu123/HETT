# Reverse human Teacher: three-epoch pilot

## Protocol

- Commit: `dedd380`
- Seed: 0
- Train / val_seen / val_unseen: first 512 trajectories per split
- Batch size: 16; gradient accumulation: 1
- Maximum action length: 20; move iteration: 10; altitude: 50 m
- Reverse Teacher weight: 0.20
- Terminal-view alignment weight / views: 0.10 / 3
- Forward Student: independent landmark encodings plus 41 x 41 target belief
- Loss normalization: active rollout steps
- Initialization: released BERT/Darknet weights; navigation and belief modules trained from scratch

This is a bounded implementation pilot, not a full-data comparison against the
released HETT Teacher.  A full run was projected to take roughly eleven hours
on the available GPU, so it was stopped before the first checkpoint and
replaced by this fixed-size pilot.

## Results

| Epoch | Total loss | Reverse loss | Forward loss | Visual align | Seen SR | Seen SPL | Seen NE | Unseen SR | Unseen SPL | Unseen NE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9431 | 0.3957 | 1.4905 | 0.002228 | 5.86 | 5.07 | 86.65 | **6.45** | **5.55** | **67.74** |
| 2 | 0.8861 | 0.3709 | 1.4013 | 0.000243 | 1.37 | 1.32 | 99.60 | 3.32 | 2.50 | 80.75 |
| 3 | 0.8671 | 0.3677 | 1.3665 | 0.000108 | 2.34 | 2.22 | 92.32 | 3.32 | 2.65 | 87.06 |

SR and SPL are percentages; NE is metres.  Peak allocated GPU memory was
48.38 GiB.  All losses and gradient norms remained finite.

## Interpretation

The implementation is trainable, but this pilot does not support replacing the
forward Teacher yet.  Both reverse and forward training losses improve while
navigation peaks after one epoch and then degrades.  The target-view cosine
loss also saturates almost immediately.  The pattern is consistent with
small-sample overfitting and an alignment objective that is too easy or can
collapse; it is not evidence that the reverse trajectory itself is harmful.

The next controlled experiment should compare forward Teacher, reverse
Teacher, joint Teachers, and a forward-Teacher warm-up on the same split.  The
visual objective should use frozen text targets plus in-batch or hard-negative
contrastive supervision, and model selection should use validation SR rather
than the training loss.

Raw metrics and checkpoints are in
`checkpoints/reverse_human_landmarks_3ep_pilot512_b16/`.
