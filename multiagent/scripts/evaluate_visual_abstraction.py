#!/usr/bin/env python3
"""Run the L0-L9 retrieval ladder after Full RGB Gate 1 has passed."""
import argparse
import json
from pathlib import Path
from evaluate_full_goal_retrieval import main as evaluate_main


if __name__ == "__main__":
    evaluate_main()

