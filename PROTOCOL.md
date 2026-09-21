# Token-level landmark grounding protocol

## Question

Can explicit instruction-token to landmark-name grounding improve the factorized dense
heatmap on unseen city blocks without changing geometry, navigation, or the action head?

## Single intended change

Keep experiment 28's contours, direction/distance bases, spatial refiner, data, loss,
six epochs, and three seeds. Replace the small marked-sentence GRU with frozen pristine
`bert-base-uncased` tokens. Each landmark name queries the instruction tokens through
cross-attention before predicting its direction and distance weights.

No RGB, target contour, target coordinate, validation label, or trajectory is an
inference input. Run only train_seen, val_seen, and val_unseen.

## Decision

Primary comparison is experiment 28: val_unseen Top-1 Hit@20 27.81% and Top-5
Recall@20 57.20%. Advance only if three-seed unseen Top-1 improves by at least 3 points
without lower Top-5, or Top-5 reaches at least 65% without lower Top-1.
