#!/usr/bin/env python3
"""Entrypoint reserved for the controlled single-factor image ablations."""
import sys
from evaluate_full_goal_retrieval import main


if __name__ == "__main__":
    if "--ablation" not in sys.argv:
        sys.argv.append("--ablation")
    main()
