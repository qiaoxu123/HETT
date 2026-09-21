# Pristine BERT token-mean selector protocol

## Question

The 20-epoch HETT BERT output is collapsed across instructions. Does the untouched
`bert-base-uncased` representation recover geometric candidate selection?

## Single intended change

Keep experiment 31's marked instructions, masked token mean, frozen encoder, candidate
set, selector, loss, splits, six epochs, and seeds. Load the original cached
`bert-base-uncased` weights instead of HETT's fine-tuned language state. No RGB, target
contour, validation label, or trajectory is an inference input.

## Decision

Pass if three-seed val_unseen reaches Top-1 Hit@20 ≥35%, Top-5 Recall@20 ≥70%, and
the seen-minus-unseen Top-1 gap is at most15 points. Compare with HETT token mean
22.69/52.55, GRU20.95/54.41, and dense geometry27.73/58.00.
