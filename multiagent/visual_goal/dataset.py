from __future__ import annotations

import json
import hashlib
from pathlib import Path

from multiagent.visual_goal.template_builder import build_episode_templates


def build_dataset(data_root: Path, output_root: Path, splits=("val_seen", "val_unseen"), max_per_split=None):
    if any(s == "test_unseen" for s in splits):
        raise ValueError("test_unseen is forbidden for this experiment")
    objects = json.loads((data_root / "cityrefer" / "objects.json").read_text())
    processed = json.loads((data_root / "cityrefer" / "processed_descriptions.json").read_text())
    records=[]
    for split in splits:
        path=data_root / "original_citynav" / f"citynav_{split}.json"
        trajectories=json.loads(path.read_text())
        def stable_key(row):
            p0=row["trajectory"][0]
            return hashlib.sha1(f"{row['area']}|{row['block']}|{row['object_ids'][0]}|{row['ann_ids'][0]}|{p0[0]:.3f}|{p0[1]:.3f}".encode()).hexdigest()
        trajectories=sorted(trajectories,key=stable_key)
        if max_per_split is not None: trajectories=trajectories[:max_per_split]
        trajectories.sort(key=lambda row: (row["area"], int(row["block"])))
        seen=set()
        for row in trajectories:
            start=row["trajectory"][0]
            map_name=f"{row['area']}_block_{row['block']}"
            object_id=int(row["object_ids"][0]); desc_id=int(row["ann_ids"][0])
            ep=f"{map_name}|{object_id}|{desc_id}|{start[0]:.3f}|{start[1]:.3f}"
            if ep in seen: continue
            seen.add(ep)
            obj=objects[map_name][str(object_id)]
            # Description-derived landmark names only. No query RGB or target coordinates enter an encoder.
            descriptions=processed.get(map_name,{}).get(str(object_id),[])
            desc=descriptions[desc_id] if desc_id < len(descriptions) else {"landmarks":[]}
            records.extend(build_episode_templates(data_root,output_root,{"episode_id":ep,"split":split,
                "map_name":map_name,"target_object_id":object_id,
                "target_xy":list(row["target_positions"][-1][:2]),"landmark_names":desc.get("landmarks",[])},
                sizes=(40,80,120)))
    manifest=output_root/"metadata"/"templates.jsonl"
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text("".join(json.dumps(x)+"\n" for x in records))
    return records
