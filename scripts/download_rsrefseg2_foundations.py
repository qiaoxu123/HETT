#!/usr/bin/env python3
"""Download only the foundation-model files needed by RSRefSeg2."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "facebook/sam2.1-hiera-large",
        cache_dir=str(cache),
        allow_patterns=[
            "config.json", "model.safetensors", "preprocessor_config.json",
            "processor_config.json", "video_preprocessor_config.json",
        ],
    )
    snapshot_download(
        "google/siglip2-so400m-patch16-512",
        cache_dir=str(cache),
        allow_patterns=[
            "config.json", "model.safetensors", "preprocessor_config.json",
            "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
            "tokenizer_config.json",
        ],
    )


if __name__ == "__main__":
    main()
