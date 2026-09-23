#!/usr/bin/env python3
"""Run one RSRefSeg2 altitude evaluation under the shared single-GPU lock."""

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


def command_output(command):
    return subprocess.run(command, capture_output=True, text=True, timeout=60).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/weights/refsegrs.pth"),
    )
    parser.add_argument(
        "--rsrefseg2-root", type=Path,
        default=Path("/home/rental/20260922_1/Workspace/RSRefSeg2"),
    )
    parser.add_argument(
        "--python", type=Path,
        default=Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/venv/bin/python"),
    )
    parser.add_argument(
        "--hf-cache", type=Path,
        default=Path("/home/rental/20260922_1/Workspace/DATA/rsrefseg2/hf_cache"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument(
        "--lock-file", type=Path,
        default=Path("/home/rental/20260922_1/Workspace/hett-experiments/.gpu-validation.lock"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "source"
    snapshot.mkdir()
    shutil.copytree(root / "scripts", snapshot / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(root / "configs", snapshot / "configs")
    shutil.copy2(root / "ALTITUDE_VIEW_PROTOCOL.md", snapshot / "ALTITUDE_VIEW_PROTOCOL.md")
    shutil.copy2(args.rsrefseg2_root / "configs_RSRefSeg2/refsegrs_infer.py", snapshot / "configs/refsegrs_infer.py")

    metrics = run / "metrics"
    command = [
        str(Path(os.path.abspath(args.python))), "tools_mmseg/test.py",
        str(snapshot / "configs/altitude_rsrefseg2_eval.py"), str(args.checkpoint.resolve()),
        "--work-dir", str(run / "mmengine"), "--cfg-options",
        f"test_dataloader.batch_size={args.batch_size}",
        f"test_dataloader.num_workers={args.num_workers}",
        f"test_dataloader.persistent_workers={str(args.num_workers > 0)}",
        f"test_dataloader.dataset.data_root={args.data_root.resolve()}",
        f"test_dataloader.dataset.ann_file={args.ann_file.resolve()}",
        f"model.backbone.cache_dir={args.hf_cache.resolve()}",
        f"model.clip_vision_encoder.cache_dir={args.hf_cache.resolve()}",
        f"model.clip_text_encoder.cache_dir={args.hf_cache.resolve()}",
    ]
    if args.max_samples > 0:
        command.append(f"test_dataloader.dataset.max_samples={args.max_samples}")
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(args.hf_cache.resolve()),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "ALTITUDE_METRIC_DIR": str(metrics),
        "PYTHONPATH": f"{args.rsrefseg2_root.resolve()}:{snapshot}:{environment.get('PYTHONPATH', '')}",
    })
    write_json(run / "provenance.json", {
        "started": stamp(),
        "hett_git_head": command_output(["git", "-C", str(root), "rev-parse", "HEAD"]).strip(),
        "hett_git_status": command_output(["git", "-C", str(root), "status", "--short"]),
        "rsrefseg2_git_head": command_output(["git", "-C", str(args.rsrefseg2_root), "rev-parse", "HEAD"]).strip(),
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": digest(args.checkpoint)},
        "annotation": {"path": str(args.ann_file.resolve()), "sha256": digest(args.ann_file)},
        "command": command,
    })
    (run / "pip_freeze.txt").write_text(command_output([str(args.python), "-m", "pip", "freeze"]))
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("w") as lock:
        write_json(run / "status.json", {"time": stamp(), "phase": "waiting_gpu"})
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (run / "evaluation.log").open("w") as log:
            child = subprocess.Popen(
                command, cwd=args.rsrefseg2_root, env=environment,
                stdout=log, stderr=subprocess.STDOUT,
            )
            while child.poll() is None:
                write_json(run / "status.json", {
                    "time": stamp(), "phase": "evaluation", "pid": child.pid,
                    "log_bytes": (run / "evaluation.log").stat().st_size,
                    "gpu": command_output([
                        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ]).strip(),
                })
                time.sleep(args.interval)
            code = child.returncode
        write_json(run / "status.json", {
            "time": stamp(), "phase": "complete" if code == 0 else "failed", "exit_code": code,
        })
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
