# Instruction-defined frame diagnostic

A frame is built from instruction text and map objects before the offline target ID is read. Geometry evaluation uses the deterministic rule parser; DeepSeek is evaluated separately for language consistency. The executable parking scope is selected from a uniquely named reference or a unique parking region. If several parking regions remain, the sample abstains. The true car is never inserted into the candidate set after scope selection.

Detected frame cues: 3,117/27,045 (11.5%). Detected frame plus ordinal layout: 1,535/27,045 (5.7%). This is regex coverage, not parser precision.

DeepSeek sampled 150 instructions with two ontology orders; 141 produced valid JSON both times. Exact consistency 20.6%; frame fields 42.6%; reference phrase 63.1%; layout Jaccard 0.660; ordinal agreement 68.8%.

| Split | Parsed frame + layout | Resolved oriented frame in evaluated parking scope |
|---|---:|---:|
| train_seen | 1381 | 206 |
| val_seen | 135 | 30 |
| val_unseen | 19 | 0 |

A building footprint major/minor axis is *unoriented*. The words “long side to the right” alone do not mathematically identify which physical end of that axis is right. The implementation refuses to choose a sign from GT or an arbitrary target position. A separate located reference (“road at the top”, “building to the right”) can resolve the sign relative to the text-selected parking region. Near-square footprints are rejected at axis confidence < 0.12.

PCA and global frames are evaluated as baselines. The 180° flip and reference swap are behavioral controls; a change in score demonstrates execution sensitivity, not grounding accuracy.
