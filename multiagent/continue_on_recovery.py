"""Continue a two-epoch HETT run only if its validation SR recovers."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_NAME = "main_system_teacherfix_b16_adam_2ep"
EXPECTED_TRAIN_SAMPLES = 21877
EXPECTED_TRAIN_BATCHES = 1368
EXPECTED_VALIDATION_SAMPLES = {"val_seen": 2470, "val_unseen": 2697}


def initial_run_active(pid):
    command_line = Path(f"/proc/{pid}/cmdline")
    try:
        return RUN_NAME.encode() in command_line.read_bytes()
    except FileNotFoundError:
        return False


def epoch_sr(record, epoch):
    lines = record.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == f"epoch {epoch}")
    except StopIteration as exc:
        raise ValueError(f"epoch {epoch} validation block is missing") from exc

    results = {}
    for line in lines[start + 1:]:
        if line.startswith("BEST RESULT") or line.startswith("epoch "):
            break
        for split in EXPECTED_VALIDATION_SAMPLES:
            if line.startswith(f"{split} ,"):
                match = re.search(r"(?:^|,\s*)sr:\s*([0-9.]+)", line)
                if match is None:
                    raise ValueError(f"{split} SR is missing")
                results[split] = float(match.group(1))
    if set(results) != set(EXPECTED_VALIDATION_SAMPLES):
        raise ValueError(f"incomplete epoch {epoch} validation results: {results}")
    return results


def check_coverage(record):
    for epoch in (0, 1):
        train_line = (
            f"EPOCH_TIMING epoch={epoch} "
        )
        matching = [line for line in record.splitlines() if line.startswith(train_line)]
        if len(matching) != 1:
            raise ValueError(f"epoch {epoch} training coverage record is missing or duplicated")
        if (
            f"train_samples={EXPECTED_TRAIN_SAMPLES}" not in matching[0]
            or f"train_batches={EXPECTED_TRAIN_BATCHES}" not in matching[0]
        ):
            raise ValueError(f"epoch {epoch} training coverage differs: {matching[0]}")
        for split, count in EXPECTED_VALIDATION_SAMPLES.items():
            expected = f"VALIDATION_TIMING epoch={epoch} split={split} "
            matching = [line for line in record.splitlines() if line.startswith(expected)]
            if len(matching) != 1 or f"samples={count}" not in matching[0]:
                raise ValueError(f"epoch {epoch} {split} validation coverage differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--seen-min", type=float, default=26.60)
    parser.add_argument("--unseen-min", type=float, default=13.76)
    args = parser.parse_args()

    checkpoint_dir = REPO_ROOT / "checkpoints" / RUN_NAME
    initial_log = REPO_ROOT / "logs" / f"{RUN_NAME}.log"
    print(f"Watching {RUN_NAME} pid={args.pid}", flush=True)
    while initial_run_active(args.pid):
        time.sleep(args.poll_seconds)

    record = (checkpoint_dir / "train.txt").read_text()
    log = initial_log.read_text()
    if any(error in log for error in ("Traceback (most recent call last)", "CUDA out of memory", "KeyboardInterrupt")):
        raise RuntimeError("initial two-epoch training ended with an error; not continuing")

    config = json.loads((checkpoint_dir / "training_args.json").read_text())
    expected_config = {"seed": 0, "grid_size": 5, "batch_size": 16, "optim": "adam", "epochs": 2}
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(f"initial training configuration differs: {config}")

    check_coverage(record)
    results = epoch_sr(record, 1)
    print(f"Epoch 1 SR: {results}; thresholds: seen={args.seen_min}, unseen={args.unseen_min}", flush=True)
    if results["val_seen"] < args.seen_min or results["val_unseen"] < args.unseen_min:
        print("Recovery threshold not met; training stops after two epochs.", flush=True)
        return

    checkpoint = checkpoint_dir / "latest"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"two-epoch checkpoint missing: {checkpoint}")
    snapshot = checkpoint_dir / "epoch2_before_continuation"
    shutil.copy2(checkpoint, snapshot)
    print(f"Recovery threshold met. Preserved {snapshot}; resuming epochs 2 through 19.", flush=True)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0", "RUN_NAME": RUN_NAME, "EPOCHS": "20",
        "BATCH": "16", "GRID": "5", "SEED": "0",
    })
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    resume_log = REPO_ROOT / "logs" / f"{RUN_NAME}_resume_to20.log"
    command = [
        "bash", "multiagent/train_1gpu.sh", "--optim", "adam",
        "--checkpoint", str(checkpoint), "--resume_optimizer",
    ]
    with resume_log.open("w") as output:
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    print(f"Continuation exited with code {result.returncode}; log: {resume_log}", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
