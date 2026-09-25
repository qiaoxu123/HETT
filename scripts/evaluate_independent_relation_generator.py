#!/usr/bin/env python3
"""Evaluate a HETT-free geometry generator followed by relation reranking."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multiagent.models.relation_goal import RelationGoalRanker
from multiagent.multi_landmark import MultiLandmarkIndex


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-artifact", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def summarize(distance, rank, landmark_count):
    row = torch.arange(len(distance))
    selected = distance[row, rank]

    def group(mask):
        values = selected[mask]
        return {
            "count": int(mask.sum()),
            "hit20": float(values.le(20).float().mean().mul(100)),
            "mean_ne": float(values.mean()),
            "median_ne": float(values.median()),
        }

    return {
        **group(torch.ones(len(distance), dtype=torch.bool)),
        "multi_landmark": group(landmark_count.ge(2)),
    }


@torch.inference_mode()
def score_mode(models, index, episode_ids, candidates, device, batch_size, control):
    chunks = []
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        landmarks = index.batch(episode_ids[start:end], device, control=control)
        points = candidates[start:end].to(device)
        values = [model(
            points, landmarks.centers, landmarks.boundaries, landmarks.valid,
            landmarks.relation, landmarks.cue, use_relation=True,
        ) for model in models]
        chunks.append(torch.stack(values).mean(0).cpu())
    return torch.cat(chunks)


def main():
    args = parse_args()
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines()
            if line.strip()]
    candidates = torch.tensor([row["candidates"] for row in rows], dtype=torch.float32)
    targets = torch.tensor([row["target"] for row in rows], dtype=torch.float32)
    distance = torch.linalg.vector_norm(
        candidates - targets[:, None], dim=-1) * 410.0

    index = MultiLandmarkIndex(args.features)
    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    split = next(
        name for name, value in payload["splits"].items()
        if len(value["episode_ids"]) == len(rows)
        and all(value["episode_ids"][i][0] == rows[i]["map_name"]
                for i in range(len(rows)))
    )
    episode_ids = payload["splits"][split]["episode_ids"]
    landmark_count = payload["splits"][split]["valid"].sum(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for checkpoint in args.checkpoints:
        model = RelationGoalRanker().to(device)
        model.load_state_dict(torch.load(
            checkpoint, map_location="cpu", weights_only=True))
        models.append(model.eval())

    normal = score_mode(
        models, index, episode_ids, candidates, device, args.batch_size, "normal")
    wrong_relation = score_mode(
        models, index, episode_ids, candidates, device, args.batch_size,
        "wrong_relation")
    wrong_landmark = score_mode(
        models, index, episode_ids, candidates, device, args.batch_size,
        "wrong_landmark")

    result = {
        "split": split,
        "count": len(rows),
        "anti_leakage": {
            "candidate_generator_inputs": "landmark geometry and language only",
            "ranker_inputs": "five generated candidates, landmark geometry, relation language",
            "target_use": "metrics only after all scores are computed",
            "hett_goal_logits": False,
            "rgb": False,
        },
        "generator_top1": summarize(
            distance, torch.zeros(len(rows), dtype=torch.long), landmark_count),
        "generator_oracle_top5": summarize(
            distance, distance.argmin(-1), landmark_count),
        "generator_plus_relation_ranker": summarize(
            distance, normal.argmax(-1), landmark_count),
        "wrong_relation_control": summarize(
            distance, wrong_relation.argmax(-1), landmark_count),
        "wrong_landmark_control": summarize(
            distance, wrong_landmark.argmax(-1), landmark_count),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.goal_artifact:
        chosen = candidates[torch.arange(len(candidates)), normal.argmax(-1)]
        args.goal_artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": 1,
            "episode_ids": episode_ids,
            "goals": chosen.half(),
            "anti_leakage": {
                "included": "episode id and independently predicted goal",
                "excluded": "target coordinate, HETT goal logits, RGB, success labels",
            },
        }, args.goal_artifact)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
