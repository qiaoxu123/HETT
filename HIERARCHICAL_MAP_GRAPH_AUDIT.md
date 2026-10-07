# Phase 1: what the map becomes

Branch `2027-CVPR/hierarchical-spatial-graph`, on `f79f64f`. Nothing is trained
here and no instruction is read. Source: `scripts/audit_map_graph.py` →
`artifacts/spatial_graph/map_graph_audit.json`, over all 34 blocks.

## Why the map is rebuilt at all

The previous round ended with a structural finding it could not act on: a
landmark phrase resolves to a *set* of entities rather than to one, because a
road is annotated as many small pieces that all carry the same name. Every
downstream arm then had to guess which piece the sentence meant, and the audit
showed there is nothing in the instruction to guess with.

The fix is to stop making the guess. A road name denotes a road, so the graph
has a **RoadRegion** node — the segments sharing that name that are also
spatially connected — and language binds to the region.

## A correction carried in from the previous round

Two of the last round's reports said `aldridge road` was 180 separate entities,
and quoted 103 and 88 for two other names. **Those figures were counts of
samples, not of entities** — they came from a table whose rows were samples, and
they overstate the group size by up to twentyfold. The true maximum, measured
here, is **15 segments sharing one name in a block** (`walsall road`,
birmingham_block_5); `aldridge road` reaches 9.

The finding itself survives: 41.6% of anchor phrases still name more than one
entity, the median ambiguous phrase still resolves to 3 entities, and the
maximum is 18. But the illustration of it was wrong, and
`REFERENCE_BINDING_AUDIT_V2.md` now carries the correction.

## The graph

| | |
|---|---:|
| blocks | 34 |
| nodes | **27,356** |
| buildings | 2,591 |
| landmarks (named, non-building, non-road) | 53 |
| objects (unnamed, non-building, non-road) | 21,958 |
| road segments | 1,672 |
| **road regions** | **486** |
| intersections | 596 |

Edge schema: **26 columns**, one definition in `edge_features.EDGE_FEATURES`,
indexed through `EDGE_INDEX` everywhere. Verified finite on 170 sampled edges of
real geometry, not only on the synthetic cases in the tests.

### Road regions

| | |
|---|---:|
| regions | 486 |
| mean segments per region | 1.48 |
| **median** | **1** |
| max | 9 |
| p90 | 3 |
| named segments inside a multi-segment region | 48.5% |

Half of all named road segments are in a region that groups two or more of them,
so the hierarchy is load-bearing for them rather than a renaming. The median of
1 says the rest are single-piece roads, which is a property of the annotation
rather than a failure of the grouping.

**59 names split into more than one region.** That is the check that matters:
grouping by name alone would have merged physically separate roads wherever a
name is reused, and the connectivity requirement prevents it. Without that
requirement the largest region in the corpus would be a 66-segment blob.

### Two decisions worth recording

**Unnamed segments do not become regions.** The first version grouped them like
any other, by adjacency alone, which merged each block's entire unnamed road
network into one blob of up to 66 segments — a connected component, not a road,
and not something a sentence can name. They remain segment nodes with their
geometry and their edges; they simply have no region to bind to. The corpus has
954 unnamed against 718 named road segments, so this is not a small corner.

**Footpaths, rails and bridges are not roads.** The corpus's road language is
about drivable road, and folding a parallel footpath into "Aldridge Road" would
make the region a worse approximation of the thing the sentence means.

## What a node may not hold

`target flag`, `candidate index`, `GT rank` and `referenced flag` are absent from
`Node` by construction. Each of the last three rounds lost time to an arm that
had quietly read the answer; the cheapest defence is for the object to have
nowhere to put it, and a test asserts the fields do not exist rather than
trusting the builder not to fill them.

The graph is built from the **whole block**, never from a candidate list. That
is the mistake the feature round made and diagnosed: a relation block over the
candidate list scored 0.685 with 3.8k parameters and no scene information,
because the list is drawn around the answer.

## Phase 1 verdict

**PASS.** The graph is the entity set the rest of the round needs: names denote
regions, regions carry real geometry, the edge schema has one definition and no
infinities, and nothing in it knows which entity is the answer.

## Reproducing

```bash
PY=/home/rental/20260922_1/miniconda3/envs/AirVLN39/bin/python
$PY -m pytest tests/test_spatial_graph.py -q    # 16 tests
$PY scripts/audit_map_graph.py                  # ~90 s
```
