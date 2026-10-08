# HETT selector optimization informed by SBF

## Evidence and scope

Direct frozen SBF selector transfer increased seen SR 27.34 -> 40.63%, but reduced unseen SR 25.39 -> 19.14% and SPL 20.28 -> 10.97% (fixed 256/split). This does not establish which input or module causes the regression. The separate lightweight HETT reranker has consumed its two permitted training epochs. No further training is started here.

## First diagnostic: bounded semantic correction

Retain the transferred SBF selector's token-level language, candidate RGB/SigLIP, agent geometry and separate landmark name+geometry inputs. Preserve HETT heatmap evidence at the final ranking stage:

```
score = z(heatmap candidate logits) + 0.25 * tanh(z(SBF selector logits))
```

`z` standardizes over the same 20 candidates. Standardization does not make these calibrated probabilities. The correction is bounded in [-0.25,0.25]; therefore a standardized heatmap Top-1 margin >0.5 cannot be overturned. Smaller margins may be reranked by semantic/geometric evidence. Equal/constant scores are finite, and candidate permutation does not change the mathematical operation.

This is an experimental inference-only fusion, not a retrained HETT selector and not a modification to the SBF reproduction baseline. Weight 0.25 is fixed before this run, not selected by a validation sweep. No labels enter scoring; GT remains evaluator-only. This may still worsen results: bounded changes are not guaranteed to be correct. No gain is claimed before paired evaluation.

## Evaluation

Fixed seed 0, val_seen/unseen 256 each, exactly the existing paired episode IDs, bounded 50m controller, 20-step horizon, radius 20m, stagnation 5. Compare plain heatmap Top-1 to bounded fusion. Direct selector transfer is a separately completed reference. No Test-Unseen. No new optimizer steps. Report SR/SPL/OSR/NE and initial selected Hit@20, with paired delta-SR confidence intervals. Nine focused tests passed before launch.

```
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/20260922_1/miniconda3/envs/AirVLN39/bin/python scripts/run_selector_fusion.py
```

Results: `artifacts/bounded_sbf_fusion/results.json` and `.csv`; source and checkpoint hashes recorded in manifests. Baseline trajectory parity is checked against the completed direct-transfer experiment.

## Subsequent learned-selector design (not trained in this run)

If a new training budget is authorized, train on HETT-generated candidates and policy states. Preserve full instruction tokens instead of only mean-pooled BERT; represent each landmark with its own identity+geometry token; use geometry at the **refined executed waypoint**, rather than cell centers while supervising refined coordinates. Candidate observation masks should distinguish absent visual evidence. Retain an explicit field-confidence path, with ablations to measure its utility. Evaluate selection error and closed-loop metrics separately; a stronger selector alone cannot guarantee correct stopping.

The already existing semantic-geometric selector worktree is a source for reusable token architecture, not proof of effectiveness or an automatically compatible checkpoint. Keep the current repaired baseline until a paired unseen evaluation demonstrates improvement without a severe SPL loss.
