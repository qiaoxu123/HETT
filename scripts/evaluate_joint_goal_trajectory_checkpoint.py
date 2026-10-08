#!/usr/bin/env python3
"""Re-evaluate configured policies on an already saved joint-run checkpoint."""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
START_DIR = Path.cwd()
sys.path[:0] = [str(ROOT / "multiagent"), str(ROOT), str(ROOT / "scripts")]
from run_joint_goal_trajectory_experiment import clean, dump, new_agent, evaluate, resolve_config_path
from parser import parse_args
from env import CityNavBatch
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["C"])
    ap.add_argument("--splits", nargs="+", default=None)
    opt = ap.parse_args()
    config_path = opt.config if opt.config.is_absolute() else START_DIR / opt.config
    output_path = opt.output if opt.output.is_absolute() else START_DIR / opt.output
    checkpoint_path = opt.checkpoint if opt.checkpoint.is_absolute() else START_DIR / opt.checkpoint
    cfg = json.loads(config_path.read_text())
    cfg['initial_checkpoint'] = str(resolve_config_path(cfg['initial_checkpoint']).resolve())
    cfg['base_arguments'] = str(resolve_config_path(cfg['base_arguments']).resolve())
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint_payload["vln_model"]["epoch"])
    del checkpoint_payload
    out = output_path.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if list(out.iterdir()):
        raise RuntimeError(f"retry output must be empty: {out}")
    sys.argv = ["joint_selector_retry"]
    args = parse_args()
    for key, value in json.loads(Path(cfg["base_arguments"]).read_text()).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.mode = "train"
    args.resume_optimizer = False
    args.batch_size = cfg["batch_size"]
    args.seed = cfg["seed"]
    args.benchmark_batches = 0
    args.epochs = cfg["epochs"]
    args.heatmap_trajectory_enabled = True
    args.trajectory_use_for_control = False
    for key, value in cfg.items():
        if key.startswith("trajectory_"):
            setattr(args, key, value)
    args.trajectory_goal_k = cfg.get("trajectory_goal_k", 20)
    args.trajectory_selector_mode = "prior"
    args.trajectory_disable_learned_stop = False
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    agent, _ = new_agent(args, cfg)
    agent.load(str(checkpoint_path.resolve()))
    selected_splits = opt.splits or cfg["evaluation_splits"]
    envs = {
        split: CityNavBatch(split, args, batch_size=args.batch_size,
                            seed=args.seed, rank=0, world_size=1)
        for split in selected_splits
    }
    retry_cfg = dict(cfg)
    retry_cfg["variants"] = {v: cfg["variants"][v] for v in opt.variants}
    retry_cfg["evaluation_splits"] = selected_splits
    retry_cfg["code_note"] = "full checkpoint eval retry after canonicalizing source punctuation"
    dump(out / "retry_manifest.json", {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "splits": {s: len(e.data) for s, e in envs.items()},
        "variants": opt.variants,
        "config": retry_cfg,
        "started_unix": time.time(),
    })
    rows = evaluate(agent, envs, retry_cfg, out, checkpoint_epoch)
    dump(out / "results.json", rows)
    dump(out / "COMPLETE.json", {"eval_runs": len(rows), "finished_unix": time.time()})
    print("RETRY_EVAL_COMPLETE", json.dumps(clean(rows)), flush=True)


if __name__ == "__main__":
    main()
