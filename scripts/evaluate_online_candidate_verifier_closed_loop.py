#!/usr/bin/env python3
"""Execute the frozen instance verifier on current RGB inside official rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from independent_nav.official_protocol import (
    OfficialProtocol, RolloutTrace, attach_metric_only_ground_truth, move_toward,
)
from independent_nav.path_local_verifier import PathLocalCandidateVerifier
from independent_nav.semantic import ProjectSemanticNet
from multiagent.env import CityNavBatch
from scripts.audit_target_type_alignment import train_probe
from scripts.evaluate_official_static_belief import (
    aligned_indices, environment_args, summarize_traces,
)
from scripts.evaluate_path_local_pointcloud_teacher import normalized_to_world
from scripts.evaluate_path_local_rgb_evidence import (
    candidate_patches, semantic_evidence,
)
from scripts.train_path_local_candidate_verifier import predict
from scripts.train_rgb_candidate_ranker import encode_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val_seen", "val_unseen"),
                        default="val_seen")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "runs/path_local_candidate_verifier/train8192/model.pt"))
    parser.add_argument("--features", type=Path, default=Path(
        "runs/path_local_candidate_verifier/features_8192.pt"))
    parser.add_argument("--records", type=Path, default=Path(
        "runs/path_local_rgb_evidence/expanded_train_records.pt"))
    parser.add_argument("--semantic-checkpoint", type=Path, default=Path(
        "runs/independent_semantic_trainseen_valseen_select/model.pt"))
    parser.add_argument("--type-features", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/hett-experiments/45-active-altitude/"
        "runs/target_type_alignment/features.pt"))
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--hf-cache", type=Path, default=Path(
        "/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/path_local_candidate_verifier/train8192/strict_online_val_seen.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--semantic-patch-size", type=int, default=128)
    parser.add_argument("--patch-side-m", type=float, default=40.0)
    parser.add_argument("--evidence-radius-m", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def inference_plans(records, features, predicted_types, cached_ranks):
    plans = []
    for index, (record, predicted_type, cached_rank) in enumerate(zip(
            records, predicted_types, cached_ranks)):
        plans.append({
            "modes": np.asarray(record["modes"], dtype=np.float64),
            "prior": features["prior"][index],
            "text": features["text"][index],
            "geometry": features["geometry"][index],
            "relation": features["relation"][index],
            "valid": features["valid"][index],
            "predicted_type": int(predicted_type),
            "observation_action_step": int(record["observation_action_step"]),
            "cached_rank": int(cached_rank),
        })
    return plans


@torch.inference_mode()
def select_online(observations, plans, indices, verifier, encoder, processor,
                  semantic, args, device):
    patches, owners = [], []
    for local_index in indices:
        observation, plan = observations[local_index], plans[local_index]
        modes_world = np.stack([
            normalized_to_world(mode, observation["map_name"])
            for mode in plan["modes"]])
        values = candidate_patches(
            observation["rgb"], observation["pose"], modes_world,
            args.view_size, args.patch_size, args.patch_side_m)
        patches.extend(values)
        owners.extend([(local_index, 0), (local_index, 1)])
    image_vectors = encode_images(
        encoder, processor, patches, device).reshape(len(indices), 2, -1)
    semantic_patches = np.stack([
        cv2.resize(value, (args.semantic_patch_size, args.semantic_patch_size),
                   interpolation=cv2.INTER_AREA) for value in patches])
    tensor = torch.from_numpy(semantic_patches).permute(0, 3, 1, 2)
    probability = semantic(
        tensor.float().to(device).div(255.0)).softmax(1).cpu().numpy()
    radius = (args.evidence_radius_m / args.patch_side_m
              * args.semantic_patch_size)
    evidence = []
    for row, local_index in enumerate(indices):
        target_class = plans[local_index]["predicted_type"] + 1
        evidence.append([
            semantic_evidence(probability[row * 2 + rank], target_class, radius)
            for rank in range(2)])
    evidence = torch.tensor(evidence)

    def stack(name):
        return torch.stack([plans[index][name] for index in indices]).to(device)
    scores = verifier(
        image_vectors.to(device), stack("text"), stack("geometry"),
        stack("relation"), stack("valid"), stack("prior"), evidence.to(device))
    return scores.argmax(-1).cpu().tolist()


def rollout_online(env, plans, verifier, encoder, processor, semantic,
                   args, device, protocol=OfficialProtocol()):
    observations = env._get_obs(random_direction=False)
    poses = [item["pose"] for item in observations]
    traces = [RolloutTrace([pose]) for pose in poses]
    selected = [None] * len(plans); ended = np.zeros(len(plans), dtype=bool)
    for outer_step in range(protocol.max_action_len):
        ready = [index for index, plan in enumerate(plans)
                 if (not ended[index] and selected[index] is None
                     and outer_step >= plan["observation_action_step"])]
        if ready:
            values = select_online(
                observations, plans, ready, verifier, encoder, processor,
                semantic, args, device)
            for index, rank in zip(ready, values):
                selected[index] = rank
        for index, observation in enumerate(observations):
            if ended[index]:
                continue
            trace, plan = traces[index], plans[index]
            trace.action_count += 1
            if selected[index] is None:
                normalized = plan["modes"].mean(0); may_stop = False
            else:
                normalized = plan["modes"][selected[index]]; may_stop = True
            destination = env.unnormalize_position(
                normalized, observation["map_name"], protocol.map_size_m)
            if may_stop and poses[index].xy.dist_to(destination) <= 5.0:
                ended[index] = True; trace.stop_step = outer_step + 1
                continue
            poses[index] = move_toward(
                poses[index], destination, protocol.move_iteration)
            trace.trajectory.append(poses[index]); trace.movement_actions += 1
        if ended.all() or outer_step + 1 == protocol.max_action_len:
            break
        observations = env._get_obs(poses, random_direction=False)
        for index, is_ended in enumerate(ended):
            if not is_ended:
                traces[index].online_observations += 1
    return traces, selected


def main():
    args = parse_args(); protocol = OfficialProtocol(); protocol.validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_features = torch.load(
        args.features, map_location="cpu", weights_only=False)
    features = (loaded_features[args.split]
                if args.split in loaded_features else loaded_features)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    verifier = PathLocalCandidateVerifier(int(checkpoint["vision_dim"])).to(device)
    verifier.load_state_dict(checkpoint["model"]); verifier.eval()
    cached_rank = predict(verifier, features, device).argmax(-1).tolist()
    loaded_records = torch.load(
        args.records, map_location="cpu", weights_only=False)
    records = (loaded_records[args.split]
               if isinstance(loaded_records, dict) else loaded_records)
    if args.max_episodes:
        records = records[:args.max_episodes]
        cached_rank = cached_rank[:args.max_episodes]
        features = {key: (value[:args.max_episodes]
                          if isinstance(value, torch.Tensor) else value[:args.max_episodes])
                    for key, value in features.items()}

    type_features = torch.load(
        args.type_features, map_location="cpu", weights_only=True)
    type_model = train_probe(
        type_features["text"]["train_seen"],
        type_features["text"]["val_seen"], args.seed, 40, device)
    predicted_type = type_model(
        type_features["text"][args.split]["features"].float().to(device)
    ).argmax(-1).cpu().numpy()[:len(records)]
    plans = inference_plans(records, features, predicted_type, cached_rank)

    processor = AutoImageProcessor.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True)
    encoder = AutoModel.from_pretrained(
        args.model, cache_dir=str(args.hf_cache), local_files_only=True,
        torch_dtype=torch.bfloat16).to(device).eval()
    semantic = ProjectSemanticNet().to(device)
    semantic.load_state_dict(torch.load(
        args.semantic_checkpoint, map_location="cpu", weights_only=True))
    semantic.eval()

    env = CityNavBatch(
        args.split, environment_args(protocol, args.batch_size, args.max_episodes),
        batch_size=args.batch_size, seed=0, rank=0, world_size=1)
    results, traces_all, selected_all, cached_all, start = {}, [], [], [], 0
    for _ in env:
        indices = aligned_indices(start, args.batch_size, len(plans))
        batch_plans = [plans[index] for index in indices]
        traces, selected = rollout_online(
            env, batch_plans, verifier, encoder, processor, semantic,
            args, device, protocol)
        for item in attach_metric_only_ground_truth(env, traces):
            results[item["instr_id"]] = item
        count = min(args.batch_size, len(plans) - start)
        traces_all.extend(traces[:count]); selected_all.extend(selected[:count])
        cached_all.extend([plan["cached_rank"] for plan in batch_plans[:count]])
        start += args.batch_size
        if start >= len(plans):
            break
    metrics, _ = env.eval_metrics(results)
    result = {
        "experiment": "strict online path-local instance verifier",
        "split": args.split, "count": len(results),
        "official_metrics": metrics,
        "rollout_diagnostics": summarize_traces(
            traces_all, protocol.max_action_len),
        "online_vs_cached_rank_match_percent": float(np.mean(
            np.asarray(selected_all) == np.asarray(cached_all)) * 100),
        "anti_leakage": {
            "visual_inference": "inside rollout on current obs[rgb]",
            "target_in_plan_or_rollout": False, "pointcloud_at_runtime": False,
            "val_unseen_loaded": args.split == "val_unseen",
            "val_unseen_used_for_selection": False,
            "test_unseen_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
