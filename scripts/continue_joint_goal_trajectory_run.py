#!/usr/bin/env python3
"""Resume five epochs from a gated epoch-1 joint-goal checkpoint."""
import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
START_DIR = Path.cwd()
sys.path[:0] = [str(ROOT / "multiagent"), str(ROOT), str(ROOT / "scripts")]
from run_joint_goal_trajectory_experiment import clean, dump, sha, new_agent, evaluate, resolve_config_path
from parser import parse_args
from env import CityNavBatch
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--rng-state", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    opt = ap.parse_args()
    resolve = lambda path: path if path.is_absolute() else START_DIR / path
    config_path = resolve(opt.config).resolve()
    checkpoint_path = resolve(opt.checkpoint).resolve()
    rng_path = resolve(opt.rng_state).resolve()
    out = resolve(opt.output).resolve()
    cfg = json.loads(config_path.read_text())
    cfg['initial_checkpoint'] = str(resolve_config_path(cfg['initial_checkpoint']).resolve())
    cfg['base_arguments'] = str(resolve_config_path(cfg['base_arguments']).resolve())
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    start_epoch = int(checkpoint_payload["vln_model"]["epoch"])
    del checkpoint_payload
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"continuation output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    sys.argv = ["joint_goal_trajectory_continue"]
    args = parse_args()
    for key, value in json.loads(Path(cfg["base_arguments"]).read_text()).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.mode = "train"
    args.resume_optimizer = True
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
    # On-policy student rollouts must use the same trajectory controller
    # and selector as validation, rather than legacy two-stage control.
    train_policy = cfg["variants"][cfg["first_epoch_gate"]["variant"]]
    for key, value in train_policy.items():
        setattr(args, key, value)

    agent, loadreport = new_agent(args, cfg)
    agent.load(str(checkpoint_path))
    train = CityNavBatch("train_seen", args, batch_size=args.batch_size,
                         seed=args.seed, rank=0, world_size=1)
    envs = {
        split: CityNavBatch(split, args, batch_size=args.batch_size,
                            seed=args.seed, rank=0, world_size=1)
        for split in cfg["evaluation_splits"]
    }
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["cuda"])

    dump(out / "manifest.json", {
        "config": cfg,
        "resumed_from": str(checkpoint_path),
        "resumed_checkpoint_sha256": sha(checkpoint_path),
        "completed_epoch_at_resume": start_epoch,
        "rng_state": str(rng_path),
        "train_count": len(train.data),
        "validation_counts": {s: len(e.data) for s, e in envs.items()},
        "load_report": loadreport,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_unix": time.time(),
    })
    (out / "checkpoints").mkdir(exist_ok=True)
    all_rows = []
    if start_epoch >= cfg["epochs"]:
        raise ValueError(
            f"checkpoint epoch {start_epoch} already reaches configured total "
            f"{cfg['epochs']}"
        )
    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        agent.env = train
        agent.env_name = "train_seen"
        agent.logs = defaultdict(list)
        agent.experiment_step_callback = None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        agent.train(DataLoader(train, batch_size=1), 1, feedback="student")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        losses = {k: float(np.mean(v)) for k, v in agent.logs.items()
                  if "loss" in k and len(v)}
        if not all(math.isfinite(v) for v in losses.values()):
            raise RuntimeError(f"nonfinite epoch {epoch} loss")
        checkpoint = out / "checkpoints" / f"epoch{epoch:02d}.pt"
        temporary = checkpoint.with_suffix(".tmp")
        agent.save(epoch - 1, str(temporary))
        temporary.replace(checkpoint)
        torch.save({
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        }, out / "checkpoints" / f"epoch{epoch:02d}_rng.pt")
        row = {
            "epoch": epoch, "train_seconds": elapsed,
            "episodes": len(train.data),
            "batches": math.ceil(len(train.data) / train.batch_size),
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
            "losses": losses, "training_diagnostics": {k:float(sum(agent.logs.get(k,()))) for k in (
                'trajectory_stop_positive_count',
                'trajectory_stop_supervised_count',
                'trajectory_arrival_gate_blocked',
                'trajectory_plan_steps',
                'trajectory_stop_decisions',
                'trajectory_ranking_valid_count',
                'trajectory_candidate_eval_count',
                'landmark_refs_total',
                'landmark_refs_name_matched',
                'landmark_refs_truncated')},
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
        }
        dump(out / f"train_epoch{epoch:02d}.json", row)
        print("TRAIN_COMPLETE", json.dumps(clean(row)), flush=True)
        if (epoch == cfg["epochs"] or
                epoch % max(1, int(cfg.get("evaluate_every", cfg["epochs"]))) == 0):
            rows = evaluate(agent, envs, cfg, out, epoch)
            all_rows.extend(rows)
            dump(out / "results.json", all_rows)
            subprocess.run([
                sys.executable, str(ROOT / "scripts/report_joint_goal_trajectory.py"),
                str(out),
            ], check=True, cwd=ROOT)
    dump(out / "COMPLETE.json", {
        "epochs_completed": cfg["epochs"], "eval_runs": len(all_rows),
        "finished_unix": time.time(),
    })


if __name__ == "__main__":
    main()
