"""Evaluate instruction-to-landmark retrieval without running navigation."""
import argparse

from multiagent.dataset.mturk_trajectory import load_mturk_trajectories
from multiagent.landmark_multimodal_map import LandmarkMultimodalMap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        default="../data/cityrefer/landmark_multimodal_map.json",
    )
    parser.add_argument(
        "--split",
        default="val_seen",
        choices=["train_seen", "val_seen", "val_unseen", "test_unseen"],
    )
    parser.add_argument("--altitude", type=float, default=50.0)
    args = parser.parse_args()

    semantic_map = LandmarkMultimodalMap.load(args.cache)
    trajectories = load_mturk_trajectories(args.split, "all", args.altitude)

    total = 0
    hit1 = 0
    hit5 = 0
    matched = 0
    for trajectory in trajectories:
        if not trajectory.descriptions:
            continue
        instruction = trajectory.descriptions[0]
        retrieved = semantic_map.retrieve_text(
            instruction, trajectory.map_name, topk=5
        )
        total += 1
        if retrieved:
            matched += 1
        ids = [record.landmark_id for record, _ in retrieved]
        target_id = int(trajectory.object_id)
        hit1 += int(bool(ids) and ids[0] == target_id)
        hit5 += int(target_id in ids)

    denom = max(total, 1)
    print(f"split={args.split} samples={total}")
    print(f"matched_instruction_rate={matched / denom:.4f}")
    print(f"Recall@1={hit1 / denom:.4f}")
    print(f"Recall@5={hit5 / denom:.4f}")


if __name__ == "__main__":
    main()
