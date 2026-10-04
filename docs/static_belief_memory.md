# Static Belief + Global Landmark Map + Navigation Memory

This branch adds a minimal three-state navigation framework without changing the existing HETT controller.

## Core states

1. Global Landmark Map: static world reference containing all named landmark contours plus the existing instruction-referenced landmark mask.
2. Belief Map B_t: dense target probability map. B_0 is produced from instruction + static map only; later perception can update it to B_t.
3. Navigation Memory H_t: compact structured history containing trajectory xy, explored mask, seen landmarks, and candidate status. Raw RGB history is deliberately excluded.

## Static belief model

Input:
- instruction token features from the existing text encoder
- static global landmark raster
- optional referenced-landmark/start-state channels

Output:
- dense belief logits
- normalized belief probabilities B_0
- spatial features for future belief-update modules

Architecture:

    static raster -> small CNN -> spatial tokens
                              |
    instruction tokens ------+-> cross-attention -> belief decoder -> B_0

No RGB, depth, trajectory history, target-distance signal, or online explored-area history is required for B_0.

## Training

Use a Gaussian target centered on the ground-truth goal and soft field cross entropy. Start with a static-only experiment; do not train the controller at the same time.

Recommended ablations:
1. instruction only
2. instruction + global landmark contour
3. + instruction-referenced landmark mask
4. + start pose/yaw channels

## Top-K hypotheses

Run greedy NMS on B_0 and keep K in {1,4,8,16}. A static prior is considered useful when at least one Top-K hypothesis is near the target even if Top-1 is ambiguous.

## Verification gates

### Gate A - representation sanity

Run:

    python -m unittest tests.test_global_landmark_prior tests.test_static_belief_memory

Checks:
- global and referenced landmark channels remain independent
- B_0 sums to one
- NMS suppresses neighboring duplicate peaks
- navigation memory updates deterministically

### Gate B - offline static localization

Train only StaticBeliefModel on train_seen and evaluate val_seen / val_unseen without environment rollout.

Report:
- peak distance / NE
- Recall@K within 20 m and 40 m for K=1,4,8,16
- oracle candidate distance@K
- belief entropy

The primary metric is Top-K coverage, not Top-1 accuracy.

### Gate C - value of online evidence

After B_0 is validated, add a separate online update model:

    B_(t-1) + current RGB/depth + pose + H_t -> B_t

Keep the same Recall@K/oracle-distance metrics. Dynamic perception should improve B_t over B_0 before it is connected to the controller.

### Gate D - closed loop

Only after Gates A-C pass, connect B_t/Top-K to navigation and report SR, OSR, SPL, NE, nDTW and sDTW together with the localization metrics above.

## First experiment

Recommended first run: 1-2 epochs, batch size 16, bf16. The first question is only whether language + static landmark structure can produce a useful B_0 whose Top-K covers the true goal region.
