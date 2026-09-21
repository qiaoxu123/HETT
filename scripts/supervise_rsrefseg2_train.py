#!/usr/bin/env python3
"""Supervise CityRefer fine-tuning with source snapshots and the shared GPU lock."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def output(command):
    return subprocess.run(command, capture_output=True, text=True, timeout=60).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-ann", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--val-ann", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/tenant2/dataext/rsrefseg2/weights/refsegrs.pth"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpoint writes (intended for short smoke tests).",
    )
    parser.add_argument("--lock-file", type=Path, default=Path("/home/tenant2/Workspace/hett-experiments/.gpu-validation.lock"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "source"
    snapshot.mkdir()
    shutil.copytree(root / "scripts", snapshot / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(root / "configs", snapshot / "configs")
    shutil.copy2(root / "PROTOCOL.md", snapshot / "PROTOCOL.md")
    rs_repo = Path("/home/tenant2/Workspace/RSRefSeg2")
    shutil.copy2(rs_repo / "configs_RSRefSeg2/refsegrs_infer.py", snapshot / "configs/refsegrs_infer.py")

    python = Path("/home/tenant2/dataext/rsrefseg2/venv/bin/python")
    command = [
        str(python), "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=1",
        "tools_mmseg/train.py", str(snapshot / "configs/cityrefer_rsrefseg2_train.py"),
        "--work-dir", str(run / "training"), "--launcher", "pytorch", "--cfg-options",
        f"load_from={args.checkpoint.resolve()}",
        f"train_cfg.max_epochs={args.epochs}",
        f"train_dataloader.batch_size={args.batch_size}",
        f"train_dataloader.num_workers={args.num_workers}",
        f"train_dataloader.persistent_workers={str(args.num_workers > 0)}",
        f"train_dataloader.dataset.data_root={args.train_root.resolve()}",
        f"train_dataloader.dataset.ann_file={args.train_ann.resolve()}",
        f"val_dataloader.num_workers={args.num_workers}",
        f"val_dataloader.persistent_workers={str(args.num_workers > 0)}",
        f"val_dataloader.dataset.data_root={args.val_root.resolve()}",
        f"val_dataloader.dataset.ann_file={args.val_ann.resolve()}",
        f"optim_wrapper.optimizer.lr={args.lr}",
    ]
    if args.no_checkpoint:
        command.extend([
            "default_hooks.checkpoint.interval=999999",
            "default_hooks.checkpoint.save_best=None",
            "default_hooks.checkpoint.save_last=False",
        ])
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/home/tenant2/dataext/rsrefseg2/hf_cache",
        "HF_DATASETS_CACHE": "/home/tenant2/dataext/rsrefseg2/hf_datasets_cache",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": f"{rs_repo}:{snapshot}:{env.get('PYTHONPATH', '')}",
    })
    write_json(run / "provenance.json", {
        "started": stamp(),
        "hett_git_head": output(["git", "-C", str(root), "rev-parse", "HEAD"]).strip(),
        "hett_git_status": output(["git", "-C", str(root), "status", "--short"]),
        "rsrefseg2_git_head": output(["git", "-C", str(rs_repo), "rev-parse", "HEAD"]).strip(),
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": digest(args.checkpoint)},
        "train_annotation": {"path": str(args.train_ann.resolve()), "sha256": digest(args.train_ann)},
        "val_annotation": {"path": str(args.val_ann.resolve()), "sha256": digest(args.val_ann)},
        "command": command,
    })
    (run / "pip_freeze.txt").write_text(output([str(python), "-m", "pip", "freeze"]))
    (run / "gpu_initial.txt").write_text(output(["nvidia-smi"]))

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("w") as lock:
        write_json(run / "status.json", {"time": stamp(), "phase": "waiting_gpu"})
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.write(f"{os.getpid()} {run}\n")
        lock.flush()
        with (run / "training.log").open("w") as log:
            child = subprocess.Popen(command, cwd=rs_repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                write_json(run / "status.json", {
                    "time": stamp(), "phase": "training", "pid": child.pid,
                    "log_bytes": (run / "training.log").stat().st_size,
                    "gpu": output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]).strip(),
                    "disk_free_gib": shutil.disk_usage(run).free / 2**30,
                })
                time.sleep(args.interval)
            code = child.returncode
        checkpoints = []
        for path in sorted((run / "training").glob("*.pth")):
            checkpoints.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)})
        write_json(run / "checkpoints.json", checkpoints)
        write_json(run / "status.json", {"time": stamp(), "phase": "complete" if code == 0 else "failed", "exit_code": code})
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
