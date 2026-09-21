# Explicit geometric candidate selector protocol

## Question

The factorized candidate set has a 91.32% val_unseen Hit@20 oracle ceiling, while the
dense heatmap reaches only 27.81%. Can explicit language-based candidate selection
close that gap?

## Design fixed before the run

- Use the same per-landmark `9 directions × 6 contour distances` peaks as experiment 28.
- Add one aggregate between-landmarks candidate when at least two landmarks are named.
- Mark each landmark as `<ref>` in turn and use the same small bidirectional GRU.
- Directly score all `4 × 54 + 1 = 217` candidates. A scalar landmark gate is learned
  jointly with the 54 relation scores.
- Optimize negative log probability assigned to all candidates within 20 m of the GT;
  if none exists, supervise the nearest candidate.
- At inference, retain five metric-NMS candidates. Their probabilities can be splatted
  with 10 m Gaussians to form the target heatmap.

Inputs remain instruction, landmark names, and landmark contours only. RGB, target
contours, validation labels, and trajectory data are excluded. Train on train_seen for
six epochs and evaluate three seeds 2701/2702/2703 on val_seen and val_unseen only.

## Decision

Pass if the three-seed val_unseen mean reaches Top-1 Hit@20 ≥35%, Top-5 Recall@20
≥70%, and the seen-minus-unseen Top-1 gap is no larger than 15 points. Compare against
experiment 28 (27.81/57.20) and experiment 27 (27.73/58.00).
