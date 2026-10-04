# Pure geometry reasoner

The geometry reasoner is an offline, zero-parameter evaluator over Static B0
peaks. It accepts only instruction-derived relation tokens, referenced landmark
geometry, candidate coordinates, and a causal pose. RGB and action prediction
are absent.

Run through `scripts/supervise_experiment.py --phase geometry-reasoner` with a
Static B0 cache. All tuning is performed on val_seen. The separate visualization
phase exports auditable candidate-level component scores.

Oracle outputs are explicitly privileged diagnostics: target coordinates build
the perfect relation signature but never enter the deployable reasoner.
