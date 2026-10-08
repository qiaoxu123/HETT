#!/usr/bin/env python3
"""Export frozen HETT dense-belief Top-16 and candidate scene crops offline.

This replays only observed CityNav trajectory prefixes to create the HETT map
state. It makes one frozen Stage-1 forward pass per sampled pose and never
invokes action selection, controller updates, or navigation rollout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BertTokenizerFast

from multiagent.cityreferobject import get_city_refer_objects
from multiagent.dataset.episode import Episode
from multiagent.dataset.mturk_trajectory import MTurkTrajectory
from multiagent.defaultpaths import MTURK_TRAJECTORY_DIR
from multiagent.defaultpaths import OBJECTS_PATH
from multiagent.mapdata import MAP_BOUNDS
from multiagent.maps.landmark_nav_map import LandmarkNavMap
from multiagent.models.ET_haa import ET, greedy_nms_topk
from multiagent.models.dark_net import Darknet
from multiagent.models.vln_model import CustomBERTModel
from multiagent.space import Pose4D
from multiagent.visual_goal.abstraction import VisualMasks, make_levels
from multiagent.visual_goal.encoder import build_encoder
from multiagent.visual_goal.template_builder import (
    IMAGE_SIZE, MapRGB, _annotation_regions, _save_rgb, opaque_key, polygon_mask,
)
from multiagent.scripts.evaluate_full_goal_retrieval import _encode_image_list, _masks, _read_jsonl


def _goal_area(candidate_xy, extent_m, cell_m):
    x, y = candidate_xy
    half = cell_m / 2
    return [[x - half, y + half], [x + half, y + half],
            [x + half, y - half], [x - half, y - half]]


def _model_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint parameters: {unexpected[:5]}")
    return missing


def _load_models(args, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = json.loads(args.training_args.read_text())
    model_args = SimpleNamespace(**hparams)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    language_model = CustomBERTModel().to(device).eval()
    vision_model = Darknet(args.darknet_config, 224).to(device).eval()
    stage1 = ET(model_args).to(device).eval()
    for name, model in (("lang_model", language_model), ("vision_model", vision_model), ("vln_model", stage1)):
        if name not in ckpt:
            raise KeyError(f"B0 checkpoint lacks {name}")
        _model_state(model, ckpt[name]["state_dict"])
    return tokenizer, language_model, vision_model, stage1, model_args


def _features(image_rgb, description, pose, map_name, nav_map, tokenizer, language_model,
              vision_model, stage1, args, device):
    encoded = tokenizer([description], padding=True, return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    lang_mask = encoded["attention_mask"].to(device).bool()
    with torch.inference_mode():
        lang, lang_cls, _ = language_model(ids, lang_mask)
        bgr = np.asarray(image_rgb, dtype=np.uint8)[..., ::-1].transpose(2, 0, 1).copy().astype(np.float32)
        bgr = (bgr - np.asarray([60.134, 49.697, 40.746], np.float32)[:, None, None])
        bgr = bgr / np.asarray([29.99, 24.498, 22.046], np.float32)[:, None, None]
        frame = vision_model(torch.from_numpy(bgr[None]).to(device)).reshape(1, 1, 512, 49)
        bounds = MAP_BOUNDS[map_name]
        pos = torch.tensor([[(pose.x - bounds.x_min) / args.map_meters,
                             (bounds.y_max - pose.y) / args.map_meters]], dtype=torch.float32, device=device)
        direction = torch.tensor([[math.sin(pose.yaw), math.cos(pose.yaw), *pos[0].tolist()]], device=device)
        gs = args.grid_size
        grid_xy = torch.tensor([[(i + .5) / gs, (j + .5) / gs]
                                for i in range(gs) for j in range(gs)], dtype=torch.float32, device=device)[None]
        maps = torch.from_numpy(nav_map).float().unsqueeze(0).to(device)
        output = stage1(
            directions=direction[:, None], frames=frame, lenths=[1],
            grid_fts=torch.zeros((1, 0, 768), device=device),
            grid_index=torch.zeros((1, 0), device=device), maps=maps,
            lang=lang, lang_mask=lang_mask, candidates=grid_xy,
            centroids=torch.zeros((1, 0, 2), device=device), lang_cls=lang_cls,
        )
        logits = output[3].reshape(1, args.heatmap_grid_size, args.heatmap_grid_size)
        probs = torch.softmax(logits.flatten(1), dim=-1).reshape_as(logits)
        ids = greedy_nms_topk(probs, top_k=args.heatmap_top_k, kernel_size=args.heatmap_nms_kernel)[0]
        score = probs.flatten()[ids]
    rows = torch.div(ids, args.heatmap_grid_size, rounding_mode="floor").cpu().numpy()
    cols = (ids % args.heatmap_grid_size).cpu().numpy()
    cell_m = args.map_meters / args.heatmap_grid_size
    candidate_xy = np.stack((bounds.x_min + (cols + .5) * cell_m,
                             bounds.y_max - (rows + .5) * cell_m), axis=1)
    return candidate_xy, score.float().cpu().numpy()


def _candidate_template(map_rgb, map_objects, refs, center, extent, cell_m, size, out_dir, key):
    rgb = map_rgb.crop_goal(center, extent)
    dummy = {"id": -1}
    masks, _ = _annotation_regions(map_objects, dummy, refs, center, extent, (size, size))
    masks["target"] = polygon_mask(_goal_area(center, extent, cell_m), center, extent, size)
    img_path = out_dir / "candidate_templates" / f"{key}.jpg"
    _save_rgb(img_path, rgb)
    mask_path = img_path.with_suffix(".npz")
    np.savez_compressed(mask_path, **masks)
    return rgb, masks, img_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", nargs="+", default=["val_seen", "val_unseen"], choices=["val_seen", "val_unseen"])
    p.add_argument("--dataset-dir", type=Path, default=Path("../artifacts/visual_goal_abstraction/dataset_v2"))
    p.add_argument("--output-dir", type=Path, default=Path("../artifacts/visual_goal_abstraction/candidates"))
    p.add_argument("--checkpoint", type=Path, default=Path("../checkpoints/heatmap_densebelief28_a9d95e3_b8_adam_20e_resume20_epochfix/latest"))
    p.add_argument("--training-args", type=Path, default=Path("../checkpoints/heatmap_densebelief28_a9d95e3_b8_adam_20e_resume20_epochfix/training_args.json"))
    p.add_argument("--darknet-config", type=Path, default=Path("../weights/yolo_v3.cfg"))
    p.add_argument("--encoder", required=True)
    p.add_argument("--encoder-model-path")
    p.add_argument("--max-queries", type=int, default=512)
    p.add_argument("--minimal-level", default="L9", choices=[f"L{i}" for i in range(10)])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"missing frozen B0 dense-belief checkpoint: {args.checkpoint}")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("B0 dense-belief candidate export currently requires CUDA")
    tokenizer, language_model, vision_model, stage1, model_args = _load_models(args, args.checkpoint, device)
    objects = get_city_refer_objects()
    raw_objects = json.loads(OBJECTS_PATH.read_text())
    encoder = build_encoder(args.encoder, device=str(device), model_path=args.encoder_model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    cell_m = model_args.map_meters / model_args.heatmap_grid_size
    for split in args.split:
        raw = json.loads((MTURK_TRAJECTORY_DIR / f"citynav_{split}.json").read_text())
        episode_lookup = {}
        for ix, row in enumerate(raw):
            map_name = f"{row['area']}_block_{row['block']}"
            episode_lookup[opaque_key(split, map_name, row['object_ids'][0], row['ann_ids'][0], ix)] = (ix, row, map_name)
        queries = _read_jsonl(args.dataset_dir / f"queries_{split}.jsonl")
        if args.max_queries and len(queries) > args.max_queries:
            rng = np.random.default_rng(args.seed)
            chosen = sorted(rng.choice(len(queries), args.max_queries, replace=False).tolist())
            queries = [queries[i] for i in chosen]
        map_rgb_cache = {}
        for qi, query in enumerate(queries):
            episode_record = episode_lookup.get(query["episode_key"])
            if episode_record is None:
                continue
            _idx, row, map_name = episode_record
            object_id, ann_id = int(row["object_ids"][0]), int(row["ann_ids"][0])
            target = objects[map_name][object_id]
            poses = MTurkTrajectory(**row).trajectory
            end_i = int(query["trajectory_index"])
            pose5 = poses[end_i]
            pose = pose5.xyzyaw
            episode = Episode(target, ann_id, [p.xyzyaw for p in poses], [])
            nav_map = LandmarkNavMap.generate_maps_for_a_trajectory(
                episode, model_args.map_shape, model_args.map_pixels_per_meter, [p.xyzyaw for p in poses[:end_i + 1]])
            refs = episode.description_landmarks
            if map_name not in map_rgb_cache:
                map_rgb_cache[map_name] = MapRGB.open(Path("../data/rgbd"), map_name)
            map_rgb = map_rgb_cache[map_name]
            image = Image.open(args.dataset_dir / query["image_path"]).convert("RGB")
            candidate_xy, belief = _features(np.asarray(image), episode.target_description, pose,
                                             map_name, nav_map, tokenizer, language_model,
                                             vision_model, stage1, model_args, device)
            template_images, target_anchor_images, minimal_images, candidate_keys = [], [], [], []
            for cell_i, center in enumerate(candidate_xy):
                key = opaque_key("belief-cell", map_name, int(round(center[0] * 10)), int(round(center[1] * 10)))
                goal, masks, img_path = _candidate_template(map_rgb, raw_objects[map_name], refs,
                                                            center, 80.0, cell_m, IMAGE_SIZE,
                                                            args.output_dir, key)
                template_images.append(Image.fromarray(goal))
                target_anchor_images.append(Image.fromarray(make_levels(np.asarray(goal), masks)["L2"]))
                minimal_images.append(Image.fromarray(make_levels(np.asarray(goal), masks)[args.minimal_level]))
                candidate_keys.append(key)
            qfeat = encoder.encode([image], batch_size=1)
            full_feats = _encode_image_list(encoder, template_images, 24)
            ta_feats = _encode_image_list(encoder, target_anchor_images, 24)
            min_feats = _encode_image_list(encoder, minimal_images, 24)
            row_out = {
                "query_key": query["query_key"], "scene_key": query["scene_key"],
                "episode_key": query["episode_key"], "map_name": map_name, "split": split,
                "belief_scores": belief.tolist(), "candidate_xy": candidate_xy.tolist(),
                "full_rgb_scores": (F.normalize(torch.from_numpy(qfeat), dim=-1).numpy() @ F.normalize(torch.from_numpy(full_feats), dim=-1).numpy().T).reshape(-1).tolist(),
                "target_anchor_scores": (F.normalize(torch.from_numpy(qfeat), dim=-1).numpy() @ F.normalize(torch.from_numpy(ta_feats), dim=-1).numpy().T).reshape(-1).tolist(),
                "minimal_template_scores": (F.normalize(torch.from_numpy(qfeat), dim=-1).numpy() @ F.normalize(torch.from_numpy(min_feats), dim=-1).numpy().T).reshape(-1).tolist(),
                "candidate_template_keys": candidate_keys,
                # GT labels are written only after all visual/B0 features are computed.
                "target_xy": list(query["target_xy"]),
            }
            all_results.append(row_out)
            if (qi + 1) % 50 == 0:
                print(f"{split}: exported {qi + 1}/{len(queries)} fixed trajectory states", flush=True)
        for raster in map_rgb_cache.values():
            raster.close()
    output = args.output_dir / "b0_top16_visual_goal_candidates.jsonl"
    output.write_text("".join(json.dumps(r) + "\n" for r in all_results))
    print(json.dumps({"rows": len(all_results), "output": str(output), "split": args.split,
                      "controller_or_navigation_rollout": False, "minimal_level": args.minimal_level}, indent=2))


if __name__ == "__main__":
    main()
