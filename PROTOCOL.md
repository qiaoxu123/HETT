# HETT token-mean selector protocol

## Question

The HETT checkpoint's BERT pooler output is collapsed across 64,044 instructions. Does
masked mean pooling over the non-collapsed token sequence improve geometric selection?

## Single intended change

Keep experiment 30's checkpoint, marked instructions, candidates, selector, loss,
splits, six epochs, and seeds. Replace only `pooler_output` with the attention-mask
weighted mean of `last_hidden_state`. BERT remains frozen and embeddings are precomputed
once. No RGB, target contour, validation label, or trajectory is an inference input.

## Decision

Pass if three-seed val_unseen reaches Top-1 Hit@20 ≥35%, Top-5 Recall@20 ≥70%, and
the seen-minus-unseen Top-1 gap is at most15 points. Compare with pooler 22.69/52.69,
GRU selector20.95/54.41, and the dense geometry heatmap27.73/58.00.
