"""Queue one isolated experiment behind another without sharing the GPU."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-run', required=True)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error('a command is required after --')
    command = args.command[1:] if args.command[0] == '--' else args.command
    status_path = Path(args.wait_run).resolve() / 'status.json'
    while True:
        try:
            status = json.loads(status_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(args.interval)
            continue
        if status.get('phase') == 'complete':
            break
        exit_code = status.get('exit_code')
        if exit_code not in (None, 0):
            raise SystemExit(
                f'predecessor failed in {status.get("phase")}: exit {exit_code}')
        time.sleep(args.interval)
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == '__main__':
    main()
