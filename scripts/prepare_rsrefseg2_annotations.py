#!/usr/bin/env python3
"""Apply the RSRefSeg2 SigLIP tokenizer limit to generated annotations."""

import argparse
import json
from pathlib import Path

from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoProcessor.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir.resolve()),
        local_files_only=not args.allow_download,
    ).tokenizer
    rows = [json.loads(line) for line in args.input.open()]
    truncated = 0
    for row in rows:
        original = row["sent"]
        original_ids = tokenizer(original, add_special_tokens=True)["input_ids"]
        encoded = tokenizer(
            original, add_special_tokens=True, truncation=True, max_length=args.max_tokens,
        )["input_ids"]
        row["original_text"] = original
        row["original_token_count"] = len(original_ids)
        row["text_truncated"] = len(original_ids) > args.max_tokens
        row["sent"] = tokenizer.decode(encoded, skip_special_tokens=True).strip()
        truncated += int(row["text_truncated"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "input": str(args.input.resolve()), "output": str(args.output.resolve()),
        "samples": len(rows), "truncated": truncated, "max_tokens": args.max_tokens,
        "model": args.model,
    }, indent=2))


if __name__ == "__main__":
    main()
