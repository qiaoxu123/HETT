# Coarse-to-fine target validation

This branch tests one change at a time on top of the corrected, interaction-disabled baseline.

## Hypothesis

The released model trains a 5x5 target classifier but navigates to a separate direct XY regression. The proposed decoder instead uses the existing 5x5 scores to select Top-K global cells and predicts an in-cell offset for each selected candidate. Existing goal MSE and target CE weights are unchanged.

## Leakage boundary

- Inference receives only the same instruction, landmark raster, current RGB/pose, candidate grid and history as the baseline.
- Ground-truth target coordinates remain loss/evaluation labels only.
- Same-landmark contrast pairs use target coordinates only to construct a fixed development subset, never as model inputs.
- `test_unseen` is not loaded.

## Seed-0 screening gate

After 512-episode one-epoch fine-tuning from the completed corrected baseline:

- first coarse cell changes for at least 25% of instruction swaps;
- mean correct-instruction target advantage is positive;
- correct instruction is better on at least 55% of paired episodes.

The smoke result is a semantic-mechanism gate, not navigation-performance evidence. A full seed-0 run is allowed only if all three checks pass.

## TODO

- [x] Isolated worktree and branch.
- [x] Coarse Top-K plus learned in-cell offset decoder.
- [x] Full-graph finite/non-zero gradient tests.
- [x] Fixed same-landmark/different-target contrast set.
- [ ] Real RTX 5090 fine-tuning smoke after corrected baseline.
- [ ] Paired hard-contrast report and plot.
- [ ] Decide whether to admit a full seed-0 run.

When screening from a checkpoint, `main.py` interprets `--epochs` as the total
epoch number, not a count to add. The runner therefore requires the recorded
parent epoch and passes `parent_epoch + 1` for exactly one additional epoch.
