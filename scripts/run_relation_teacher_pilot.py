#!/usr/bin/env python3
"""Evaluate an open-weight LLM as a relation-label teacher on trusted seeds."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(
        "runs/relation_teacher_dataset/val_seen_trusted.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--package-dir", type=Path, default=Path(
        "runs/qwen_teacher/python_packages"))
    parser.add_argument("--cache-dir", type=Path, default=Path(
        "runs/qwen_teacher/hf_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/qwen_teacher/pilot"))
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def select_balanced(rows, per_class, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strict_relation"]].append(row)
    random.Random(seed).shuffle(rows)
    selected = []
    generator = random.Random(seed)
    for relation in sorted(grouped):
        values = grouped[relation]
        generator.shuffle(values)
        selected.extend(values[:per_class])
    generator.shuffle(selected)
    return selected


def parse_json(text):
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.input.read_text().splitlines()
            if line.strip()]
    selected = select_balanced(rows, args.per_class, args.seed)

    # Keep the HETT environment unchanged: import the isolated Transformers
    # installation only for this teacher process. Torch/Numpy are already
    # loaded from the navigation environment.
    sys.path.insert(0, str(args.package_dir.resolve()))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=str(args.cache_dir))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=str(args.cache_dir), torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa").eval()
    outputs = []
    for offset in range(0, len(selected), args.batch_size):
        batch = selected[offset:offset + args.batch_size]
        conversations = [[{"role": "user", "content": row["teacher_input"]}]
                         for row in batch]
        prompts = [tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            enable_thinking=False) for conversation in conversations]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=160, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
        lengths = encoded["attention_mask"].sum(-1).tolist()
        for row, tokens, length in zip(batch, generated, lengths):
            # With left padding, generated continuations start after the full
            # padded input width rather than each unpadded token count.
            text = tokenizer.decode(
                tokens[encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = parse_json(text)
            valid = parsed is not None
            relation = parsed.get("relation") if valid else None
            landmark = parsed.get("landmark") if valid else None
            outputs.append({
                "sample_id": row["sample_id"],
                "gold_relation": row["strict_relation"],
                "gold_landmark": row["landmark_name"],
                "raw_output": text,
                "parsed": parsed,
                "json_valid": valid,
                "relation_correct": bool(valid and relation == row["strict_relation"]),
                "landmark_correct": bool(
                    valid and normalize(landmark) == normalize(row["landmark_name"])),
            })
        print(f"teacher pilot {min(offset + args.batch_size, len(selected))}/"
              f"{len(selected)}", flush=True)

    valid = [row for row in outputs if row["json_valid"]]
    result = {
        "model": args.model,
        "count": len(outputs),
        "gold_distribution": dict(Counter(
            row["gold_relation"] for row in outputs)),
        "json_valid_percent": 100.0 * len(valid) / max(len(outputs), 1),
        "relation_accuracy": 100.0 * sum(
            row["relation_correct"] for row in outputs) / max(len(outputs), 1),
        "landmark_accuracy": 100.0 * sum(
            row["landmark_correct"] for row in outputs) / max(len(outputs), 1),
        "by_relation": {},
        "gate": {
            "minimum_json_valid_percent": 98.0,
            "minimum_relation_accuracy": 85.0,
            "minimum_landmark_accuracy": 98.0,
        },
        "anti_leakage": {
            "split": "val_seen trusted seeds",
            "target_coordinate_in_prompt": False,
            "candidate_distance_in_prompt": False,
            "val_unseen_loaded": False,
            "test_unseen_loaded": False,
        },
    }
    for relation in sorted(set(row["gold_relation"] for row in outputs)):
        subset = [row for row in outputs if row["gold_relation"] == relation]
        result["by_relation"][relation] = {
            "count": len(subset),
            "accuracy": 100.0 * sum(row["relation_correct"] for row in subset)
            / max(len(subset), 1),
        }
    result["gate"]["passed"] = (
        result["json_valid_percent"] >= result["gate"]["minimum_json_valid_percent"]
        and result["relation_accuracy"] >= result["gate"]["minimum_relation_accuracy"]
        and result["landmark_accuracy"] >= result["gate"]["minimum_landmark_accuracy"])
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in outputs))
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
