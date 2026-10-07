import json
import pytest

from multiagent.visual_goal.query_manifest import load_verified_queries
from multiagent.visual_goal.dataset import build_dataset


def _row(**changes):
    row={"episode_id":"e1","split":"val_seen","map_name":"m","image":"q.png","distance_to_goal_m":30,
         "renderer_source":"airsim_calibrated","renderer_alignment_verified":True,"target_object_id":1,
         "image_origin":"uav_trajectory_pose","trajectory_pose5d":[0,0,40,0,-0.5]}
    row.update(changes); return row


def test_accepts_verified_real_trajectory_observation(tmp_path):
    (tmp_path/"q.png").write_bytes(b"frame")
    manifest=tmp_path/"q.jsonl"; manifest.write_text(json.dumps(_row())+"\n")
    assert len(load_verified_queries(manifest,tmp_path))==1


@pytest.mark.parametrize("changes",[
    {"split":"test_unseen"},
    {"renderer_source":"orthophoto_heightfield"},
    {"renderer_alignment_verified":False},
])
def test_rejects_forbidden_or_unverified_query(tmp_path,changes):
    (tmp_path/"q.png").write_bytes(b"frame")
    manifest=tmp_path/"q.jsonl"; manifest.write_text(json.dumps(_row(**changes))+"\n")
    with pytest.raises(ValueError): load_verified_queries(manifest,tmp_path)


def test_rejects_episode_object_split_overlap(tmp_path):
    (tmp_path/"q.png").write_bytes(b"frame")
    rows=[_row(),_row(split="val_unseen")]
    manifest=tmp_path/"q.jsonl"; manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ValueError,match="leakage"): load_verified_queries(manifest,tmp_path)


def test_builder_rejects_test_unseen_before_reading_data(tmp_path):
    with pytest.raises(ValueError,match="test_unseen"):
        build_dataset(tmp_path,tmp_path/"out",("test_unseen",))
