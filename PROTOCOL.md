# HETT BERT geometric selector protocol

## Question

Does the language encoder from the completed 20-epoch HETT checkpoint improve the
explicit geometric selector that failed with a small GRU?

## Single intended change

- Keep experiment 29's candidate set, labels, 20 m probability loss, metric NMS,
  splits, epochs, and seeds.
- Replace the train-from-scratch GRU with frozen 768-dimensional pooler embeddings from
  `runs/hett_baseline_fixed_20260911/checkpoints/best_val_unseen`.
- Train only a small 768→256 adapter, landmark gate, and candidate heads.
- Mark the current named landmark in each instruction as "the referenced landmark" and
  other names as "another named landmark" before encoding.

No RGB, target contour, validation label, or trajectory is an inference input. BERT
embeddings are precomputed once and reused identically across seeds. The cache records
the source checkpoint hash.

## Decision

Pass if the three-seed val_unseen mean reaches Top-1 Hit@20 ≥35%, Top-5 Recall@20
≥70%, and the seen-minus-unseen Top-1 gap is no larger than 15 points. Compare with the
GRU selector's 20.95/54.41 and the best dense geometry result 27.73/58.00.
