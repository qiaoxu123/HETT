from mmengine.config import read_base

with read_base():
    from .refsegrs_infer import *

from scripts.cityrefer_metrics import CityReferGroundingMetric

custom_imports = dict(imports=["refseg", "scripts.cityrefer_metrics"], allow_failed_imports=False)

train_root = "/tmp/cityrefer_train/images"
train_ann = "/tmp/cityrefer_train/cityrefer_train_seen.jsonl"
val_root = "/tmp/cityrefer_val/images"
val_ann = "/tmp/cityrefer_val/cityrefer_val_unseen.jsonl"

common_meta = (
    "text", "category_name", "img_path", "seg_map_path", "ori_shape", "img_shape",
    "pad_shape", "scale_factor", "flip", "flip_direction", "reduce_zero_label",
)
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadSegAnnotations),
    dict(type=Resize, scale=(1024, 1024), keep_ratio=True),
    dict(type=PackSegInputs, meta_keys=common_meta),
]
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=(1024, 1024), keep_ratio=True),
    dict(type=LoadSegAnnotations),
    dict(type=PackSegInputs, meta_keys=common_meta),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(type=dataset_type, data_root=train_root, ann_file=train_ann, pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=dataset_type, data_root=val_root, ann_file=val_ann, pipeline=test_pipeline, test_mode=True),
)

train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=2, val_interval=1)
val_cfg = dict(type=ValLoop)
param_scheduler = []
optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="bfloat16",
    optimizer=dict(type="AdamW", lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=3.0, norm_type=2),
)
val_evaluator = [dict(type="RefSegIoUMetric"), dict(type=CityReferGroundingMetric)]
default_hooks["logger"] = dict(type=LoggerHook, interval=10, log_metric_by_epoch=False)
default_hooks["checkpoint"] = dict(
    type=CheckpointHook,
    by_epoch=True,
    interval=999999,
    max_keep_ckpts=1,
    save_last=False,
    save_optimizer=False,
    save_param_scheduler=False,
    save_best="CityReferGrounding/box_acc_0.25",
    rule="greater",
)
visualizer["vis_backends"] = [dict(type=LocalVisBackend)]
randomness = dict(seed=0, deterministic=False)

model["backbone"]["cache_dir"] = "/home/tenant2/dataext/rsrefseg2/hf_cache"
model["clip_vision_encoder"]["cache_dir"] = "/home/tenant2/dataext/rsrefseg2/hf_cache"
model["clip_text_encoder"]["cache_dir"] = "/home/tenant2/dataext/rsrefseg2/hf_cache"
