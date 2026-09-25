# Reverse human Teacher with landmark decomposition

This experiment replaces the released forward Teacher rollout with a
motion-consistent reversal of the recorded human trajectory.  The following
forward Student rollout starts from a fresh environment observation and keeps
the original instruction and target supervision.

The reverse branch trains direction, progress, and terminal target-view/text
alignment.  It does not train coordinate, candidate, or dense belief losses,
because its metric destination is the original route start rather than the
described landmark target.

The forward Student uses the multi-landmark belief head: parsed landmark names
and map centres remain paired and are encoded independently, while the parsed
target phrase is used by the separate terminal-view alignment objective.

Run the three-epoch experiment with:

```bash
bash multiagent/train_reverse_landmarks.sh
```

For a bounded smoke test, append `--max_episodes 4 --max_action_len 2`.
