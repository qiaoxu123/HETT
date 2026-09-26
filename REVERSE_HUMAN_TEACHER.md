# Reverse human Teacher with landmark decomposition

This experiment replaces the released forward Teacher rollout with a
motion-consistent reversal of the recorded human trajectory.  The following
forward Student rollout starts from a fresh environment observation and keeps
the original instruction and target supervision.

The reverse branch trains direction, progress, and terminal target-view/text
alignment.  It does not train coordinate, candidate, or dense belief losses,
because its metric destination is the original route start rather than the
described landmark target.

The reverse branch remains the only Teacher.  This revision deliberately does
not restore the released forward Teacher.  Instead it closes the main gaps in
the reverse setup:

- local reverse-waypoint direction and a separate original-start bearing head;
- learned forward/reverse task embeddings to separate the two objectives;
- multi-view terminal visual attention with contrastive target-text alignment;
- initial-distance progress normalization shared by both task directions;
- normalized recurrent belief fusion, confidence-weighted landmark evidence,
  and rejection of low-similarity fuzzy landmark matches;
- a goal-stability stop rule plus stage-2 replanning when the belief peak moves;
- branch-specific losses, belief/target/stop diagnostics, validation trajectory
  archives, and optimizer/RNG checkpoint state for reproducible resume.

The forward Student uses the multi-landmark belief head: parsed landmark names
and map centres remain paired and are encoded independently, while the parsed
target phrase is used by the separate terminal-view alignment objective.

Run the three-epoch experiment with:

```bash
bash multiagent/train_reverse_landmarks.sh
```

For a bounded smoke test, append `--max_episodes 4 --max_action_len 2`.
