"""Parse CityNav instructions into one compact spatial relation with local Qwen."""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
RELATIONS = ["left", "right", "north", "south", "northeast", "northwest",
             "southeast", "southwest", "near", "across", "between", "front",
             "behind", "inside", "along", "unknown"]


def descriptions(split):
    rows = json.load((ROOT / f"data/processed_citynav/citynav_{split}.json").open())
    return [description for row in rows for description in row["descriptions"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="/home/tenant2/dataext/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).to("cuda").eval()
    prefix = (
        "Classify the target's spatial relation to its named reference landmark. "
        "Answer with exactly one word from: " + " ".join(RELATIONS) + ". "
        "Examples: 'car left of the gym' -> left; 'house between church and school' -> between; "
        "'building opposite the hospital' -> across; 'shop beside the bank' -> near. Instruction: "
    )
    result = {"model": args.model, "relations": RELATIONS, "splits": {}}
    for split in ["val_seen", "val_unseen"]:
        texts = descriptions(split)
        parsed = []
        for start in range(0, len(texts), args.batch_size):
            prompts = []
            for text in texts[start:start + args.batch_size]:
                chat = [{"role": "user", "content": [{"type": "text", "text": prefix + text}]}]
                prompts.append(processor.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True))
            batch = processor(text=prompts, padding=True, truncation=True,
                              max_length=384, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=5, do_sample=False)
            for text, output in zip(texts[start:start + args.batch_size], generated):
                raw = processor.decode(output[batch.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
                match = re.search(r"\b(" + "|".join(RELATIONS) + r")\b", raw)
                parsed.append({"description": text, "relation": match.group(1) if match else "unknown",
                               "raw": raw})
            if start % (args.batch_size * 10) == 0:
                print(f"{split} {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)
        result["splits"][split] = parsed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
