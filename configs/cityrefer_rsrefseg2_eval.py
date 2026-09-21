from mmengine.config import read_base

with read_base():
    from .refsegrs_infer import *

from scripts.cityrefer_metrics import CityReferGroundingMetric

custom_imports = dict(
    imports=["refseg", "scripts.cityrefer_metrics"],
    allow_failed_imports=False,
)

data_root = "/tmp/cityrefer_rsrefseg2/images"
ann_file = "/tmp/cityrefer_rsrefseg2/cityrefer_test_unseen.jsonl"

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=(1024, 1024), keep_ratio=True),
    dict(type=LoadSegAnnotations),
    dict(
        type=PackSegInputs,
        meta_keys=(
            "text", "category_name", "img_path", "seg_map_path", "ori_shape",
            "img_shape", "pad_shape", "scale_factor", "flip", "flip_direction",
            "reduce_zero_label",
        ),
    ),
]

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_evaluator = [
    dict(type="RefSegIoUMetric"),
    dict(type=CityReferGroundingMetric),
]

model["backbone"]["cache_dir"] = "/home/tenant2/dataext/rsrefseg2/hf_cache"
model["clip_vision_encoder"]["cache_dir"] = "/home/tenant2/dataext/rsrefseg2/hf_cache"
model["clip_text_encoder"]["cache_dir"] = "/home/tenant2/dataext/rsrefseg2/hf_cache"
