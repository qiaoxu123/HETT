# Factorized relational heatmap protocol

## Question

Does explicitly factoring target direction and distance from each landmark improve the
relational heatmap from experiment 27?

## Single intended change

- Keep the same instruction marking, per-landmark identity, pair field, 64×64 output,
  loss, CNN refiner, data splits, six epochs, and seeds 2701/2702/2703.
- Replace 12 broad relation fields with two independent distributions:
  - angle: isotropic plus eight compass sectors;
  - exterior distance from the contour: 0, 15, 30, 50, 80, and 120 metres.
- Multiply the selected angle and distance fields before combining landmarks. This can
  represent, for example, "left of A, about 30 m away" directly.

No RGB, target contour, validation label, or navigation trajectory is a model input.
Development uses val_seen and val_unseen only; test_unseen remains untouched.

## Metrics and decision

Report Top-1 Hit@20, Top-5 Recall@20, median/P90 error, 20 m probability mass, and
relation-wise metrics. Compare directly with experiment 27's three-seed means:
val_seen 37.50/73.37 and val_unseen 27.73/58.00 (Top-1/Top-5).

Pass only if the three-seed val_unseen mean reaches Top-1 ≥35% and Top-5 ≥70%, with
the seen-minus-unseen Top-1 gap no larger than 15 points.
