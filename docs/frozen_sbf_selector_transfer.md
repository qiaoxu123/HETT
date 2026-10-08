# Frozen SBF selector transfer to HETT Heatmap

Question: does the trained SBFNav epoch-14 selector improve HETT's Top-20 selection without training?

## Protocol

- Branch: `2027-CVPR/hett-frozen-sbf-selector`.
- HETT: original relative-geometry experiment's completed epoch-1 best_val_unseen.
- SBF: stable original epoch-14 best_val_unseen; SHA256 `e991b4e46ebc20b9cf7a1e350a0993c75028e863b10b55df166acd4e59510885`.
- Load all 43 original selector tensors. Missing tensors may only belong to the disabled, subsequently added spatial encoder. Original SigLIP weights/preprocessing are used.
- HETT generates the same Top-20 NMS cells as the control. Refine every candidate with the existing local soft argmax, convert HETT normalized coordinates to world meters, then use SBF's original planning transform and selector geometry normalization.
- Build observed RGB from executed policy poses, never human future poses. Retain SBF's original referenced-landmark matching, name encoding, 100m crops and 8-landmark limit.
- Selector API receives episode identity, instruction, executed pose history and predicted candidate coordinates. No target coordinates, oracle distances or labels are passed. The state builder uses episode identity to resolve instruction-linked landmarks; it does not read record.target_position.
- Output selects one existing candidate. No SBF belief field, added landmark candidates, STAY candidate, altitude head, field-score fusion or trainable adapter is used.
- Both paired variants use HETT's repaired bounded 50m waypoint execution, fixed altitude, stagnation 5, horizon 20, success radius 20m and the same HETT evaluator.
- Fixed existing 256 episode IDs each for val_seen and val_unseen, seed 0. No Test-Unseen. Zero optimizer steps.
- First validate 8 episodes per split; these are only engineering smoke checks, not performance evidence.

## Comparison scope

The paired difference isolates the transferred selector **and its original input feature pipeline**. It does not isolate selector architecture from SigLIP. The original SBF navigation results are useful context but use a different proposal and altitude pipeline. Transfer success cannot be assumed: HETT's candidate distribution and observed trajectories differ from SBF's training distribution.

## Commands

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/20260922_1/miniconda3/envs/AirVLN39/bin/python scripts/diagnose_candidate_controller.py \
  --config "$PWD/configs/diagnostics/frozen_sbf_selector.json" \
  --output "$PWD/artifacts/frozen_sbf_smoke" --limit 8

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/20260922_1/miniconda3/envs/AirVLN39/bin/python scripts/diagnose_candidate_controller.py \
  --config "$PWD/configs/diagnostics/frozen_sbf_selector.json" \
  --output "$PWD/artifacts/frozen_sbf_selector"
```

`run_frozen_sbf_transfer.py SMOKE_PID` waits for the smoke check and invokes the second command, then writes CSV and checks baseline trajectory parity against the previously completed diagnostic if present.

## Outputs

`artifacts/frozen_sbf_selector/results.json` and `.csv`: SR, SPL, OSR, NE, candidate coverage, selected/refined waypoint distances, paired episode-bootstrap delta-SR confidence intervals, gained/lost episodes, runtime. These confidence intervals are episode based, not scene clustered.

Raw episode traces remain local. Original baseline files, training results and checkpoint weights are not edited.
