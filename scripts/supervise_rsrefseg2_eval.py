#!/usr/bin/env python3
"""Supervise an external RSRefSeg2 evaluation with HETT provenance and GPU lock."""

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
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/tenant2/dataext/rsrefseg2/weights/refsegrs.pth"))
    parser.add_argument(
        "--rsrefseg2-root", type=Path,
        default=Path(os.environ.get("RSREFSEG2_ROOT", "/home/tenant2/Workspace/RSRefSeg2")),
    )
    parser.add_argument(
        "--python", type=Path,
        default=Path(os.environ.get("RSREFSEG2_PYTHON", "/home/tenant2/dataext/rsrefseg2/venv/bin/python")),
    )
    parser.add_argument(
        "--hf-cache", type=Path,
        default=Path(os.environ.get("RSREFSEG2_HF_CACHE", "/home/tenant2/dataext/rsrefseg2/hf_cache")),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--interval", type=int, default=30)
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

    rs_repo = args.rsrefseg2_root.resolve()
    shutil.copy2(rs_repo / "configs_RSRefSeg2/refsegrs_infer.py", snapshot / "configs/refsegrs_infer.py")
    shutil.copy2(root / "scripts/cityrefer_metrics.py", snapshot / "configs/cityrefer_metrics.py")
    python = args.python.resolve()
    hf_cache = args.hf_cache.resolve()
    config = snapshot / "configs/cityrefer_rsrefseg2_eval.py"
    metrics = run / "metrics"
    command = [
        str(python), "tools_mmseg/test.py", str(config), str(args.checkpoint.resolve()),
        "--work-dir", str(run / "mmengine"), "--cfg-options",
        f"test_dataloader.batch_size={args.batch_size}",
        f"test_dataloader.num_workers={args.num_workers}",
        f"test_dataloader.persistent_workers={str(args.num_workers > 0)}",
        f"test_dataloader.dataset.data_root={str(args.data_root.resolve())}",
        f"test_dataloader.dataset.ann_file={str(args.ann_file.resolve())}",
        f"model.backbone.cache_dir={hf_cache}",
        f"model.clip_vision_encoder.cache_dir={hf_cache}",
        f"model.clip_text_encoder.cache_dir={hf_cache}",
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(hf_cache),
        "HF_DATASETS_CACHE": str(hf_cache / "datasets"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "CITYREFER_METRIC_DIR": str(metrics),
        "PYTHONPATH": f"{rs_repo}:{snapshot}:{env.get('PYTHONPATH', '')}",
    })
    provenance = {
        "started": stamp(),
        "hett_git_head": command_output(["git", "-C", str(root), "rev-parse", "HEAD"]).strip(),
        "hett_git_status": command_output(["git", "-C", str(root), "status", "--short"]),
        "rsrefseg2_git_head": command_output(["git", "-C", str(rs_repo), "rev-parse", "HEAD"]).strip(),
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": digest(args.checkpoint)},
        "annotation": {"path": str(args.ann_file.resolve()), "sha256": digest(args.ann_file)},
        "data_root": str(args.data_root.resolve()),
        "command": command,
        "environment": {key: env[key] for key in (
            "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "CITYREFER_METRIC_DIR", "PYTHONPATH",
        )},
    }
    write_json(run / "provenance.json", provenance)
    (run / "pip_freeze.txt").write_text(command_output([str(python), "-m", "pip", "freeze"]))
    (run / "gpu_initial.txt").write_text(command_output(["nvidia-smi"]))

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("w") as lock:
        write_json(run / "status.json", {"time": stamp(), "phase": "waiting_gpu"})
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.write(f"{os.getpid()} {run}\n")
        lock.flush()
        with (run / "evaluation.log").open("w") as log:
            child = subprocess.Popen(command, cwd=rs_repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                write_json(run / "status.json", {
                    "time": stamp(), "phase": "evaluation", "pid": child.pid,
                    "log_bytes": (run / "evaluation.log").stat().st_size,
                    "gpu": command_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]).strip(),
                    "disk_free_gib": shutil.disk_usage(run).free / 2**30,
                })
                time.sleep(args.interval)
            code = child.returncode
        write_json(run / "status.json", {"time": stamp(), "phase": "complete" if code == 0 else "failed", "exit_code": code})
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
