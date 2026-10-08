#!/usr/bin/env python3
"""Build offline GoalScene crops and query crops from real CityNav trajectories."""
import argparse
from pathlib import Path

from multiagent.visual_goal.template_builder import build_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["train_seen", "val_seen", "val_unseen"],
                        choices=["train_seen", "val_seen", "val_unseen"])
    parser.add_argument("--data-root", type=Path, default=Path("../data"))
    parser.add_argument("--image-dir", type=Path, default=Path("../data/rgbd"))
    parser.add_argument("--output-dir", type=Path, default=Path("../artifacts/visual_goal_abstraction/dataset"))
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--query-per-bin", type=int, default=1)
    args = parser.parse_args()
    for split in args.split:
        build_split(split, args.output_dir, args.data_root, args.image_dir,
                    args.max_episodes, args.query_per_bin)


if __name__ == "__main__":
    main()

