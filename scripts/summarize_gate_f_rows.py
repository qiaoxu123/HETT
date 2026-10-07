#!/usr/bin/env python3
"""Streamed schema/coverage audit for all Gate F clause rows."""
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/spatial_canonicalization_gate_f'
REQUIRED=('sample_id','split','map_id','instruction','target_type','target_position','target_footprint',
          'anchor_phrase','anchor_type','anchor_entity_id','anchor_position','anchor_footprint',
          'second_anchor_phrase','second_anchor_type','second_anchor_entity_id','second_anchor_position','second_anchor_footprint',
          'start_position','start_heading','trajectory_xyz','trajectory_heading',
          'final_heading_3','final_heading_5','final_heading_10','final_heading_20','route_heading',
          'road_region_id','road_region_name','road_nearest_point','road_tangent','road_normal','road_side_sign',
          'anchor_major_axis','anchor_minor_axis')
def main():
    counts=Counter();sources=Counter();ids=set();episodes=set()
    for line in (OUT/'extended_geometry_rows.jsonl').open():
        row=json.loads(line);counts['rows']+=1;sources[row['clause_source']]+=1
        counts['missing_schema_rows']+=int(any(k not in row for k in REQUIRED))
        counts['duplicate_sample_ids']+=int(row['sample_id'] in ids);ids.add(row['sample_id'])
        episodes.add((row['split'],row['episode_id']))
        counts['trajectory_rows']+=int(bool(row['trajectory_xyz']))
        counts['road_hypothesis_rows']+=int(any(x['type']=='road_region' for x in row['anchor_hypotheses']))
        counts['unique_road_rows']+=int(row['road_region_id'] is not None)
        counts['between_second_rows']+=int(row['relation_phrase']=='between' and bool(row['second_anchor_hypotheses']))
        counts['between_annotation_second_rows']+=int(row['relation_phrase']=='between' and row['clause_source']=='annotation' and bool(row['second_anchor_hypotheses']))
    result={'counts':dict(counts),'source_rows':dict(sources),'unique_episodes':len(episodes)}
    (OUT/'row_coverage.json').write_text(json.dumps(result,indent=2))
    print(result)
if __name__=='__main__':main()
