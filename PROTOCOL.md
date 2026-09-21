# Landmark arrival and teacher-trajectory audit

- Read-only analysis of the released 20-epoch baseline predictions on full `val_seen` and `val_unseen`.
- Development splits only; `test_unseen` is not read.
- Arrival is measured as 2D distance from the saved pose to the union of the instruction's supplied landmark contours.
- The existing progress output is fitted only on `val_seen`; its fixed threshold is then reported on `val_unseen`.
- The added runtime output is diagnostic only. It does not switch stages, stop the agent, change model inputs, or change weights.
- `landmark_arrived` means map/pose arrival. It is not evidence that RGB recognized the landmark.

