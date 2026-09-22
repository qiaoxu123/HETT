# Progress stop threshold result

Checkpoint: landmark-clean Epoch 7 `best_val_unseen`, seed 0. All arms use the
same weights, data, 20-step budget, and deterministic evaluation. `test_unseen`
was not used.

## Main metrics

| Controller | Split | SR | SPL | NE (m) | Oracle SR | Path (m) |
|---|---|---:|---:|---:|---:|---:|
| 0.95 + stage gate | seen | 32.47 | 26.86 | 36.32 | 51.13 | 228.51 |
| 0.80 + stage gate | seen | 31.17 | 27.16 | 36.80 | 47.45 | 221.20 |
| 0.80, no stage gate | seen | 31.21 | 27.72 | 36.62 | 45.95 | 216.41 |
| 0.95 + stage gate | unseen | 19.13 | 15.73 | 51.47 | 35.22 | 252.44 |
| 0.80 + stage gate | unseen | 18.09 | 15.64 | 51.89 | 32.67 | 246.92 |
| 0.80, no stage gate | unseen | 17.80 | 15.48 | 52.25 | 31.59 | 242.13 |

Relative to the current 0.95 rule, 0.80 with the stage gate changes SR by
-1.30 points on seen and -1.04 points on unseen. Removing the stage gate changes
SR by -1.26 and -1.33 points respectively.

## Why 0.80 fails

The mathematical mapping is correct: 20 m corresponds to a target progress of
0.80. The learned output is not calibrated to that mapping well enough to serve
as a hard stop decision.

| Audit | seen | unseen |
|---|---:|---:|
| ROC AUC for classifying distance <= 20 m | 0.739 | 0.693 |
| Precision at progress >= 0.80 | 43.46% | 24.44% |
| Recall at progress >= 0.80 | 31.31% | 17.63% |
| Episodes shortened by gated 0.80 | 51.82% | 44.23% |
| Shortened episodes actually inside 20 m | 37.89% | 22.30% |
| Episodes shortened by ungated 0.80 | 53.68% | 48.02% |
| Shortened episodes actually inside 20 m | 37.48% | 21.08% |

The gated 0.80 arm has 83 unseen episodes that change from success to failure
and only 55 that change from failure to success. The ungated arm has 97 losses
and 61 gains. The stage gate is mildly protective on unseen, but it is not the
main defect.

At 0.95, the head almost never recognizes a successful state: state-level
recall is 0.41% on seen and 0.03% on unseen. At 0.80 it fires much more often,
but mostly outside the success radius. Scores also range from -0.225 to 1.050,
confirming that the unconstrained MSE output has no strict probability or
distance interpretation.

## Decision

Do not replace the current 0.95 rule with a raw 0.80 threshold. The experiment
rejects both 0.80 variants. Keep the stage gate until a better arrival signal is
available.

The next model change should be a dedicated `distance <= 20 m` arrival head,
trained with a binary classification loss and evaluated for precision/recall
and calibration. Progress regression can remain as an auxiliary task. A new
stop rule should be selected on `val_seen`, then checked once on `val_unseen`;
this single-seed diagnostic is not final deployment evidence.

Machine-readable results are in `RESULTS.json`; the controlled design is in
`PROTOCOL.md`.
