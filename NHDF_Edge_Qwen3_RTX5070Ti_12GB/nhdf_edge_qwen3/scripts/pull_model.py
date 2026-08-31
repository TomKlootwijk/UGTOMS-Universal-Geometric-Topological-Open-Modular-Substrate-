#!/usr/bin/env python3
"""Download the selected Hugging Face checkpoint without loading it."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--token")
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=target,
        token=args.token,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "*.json",
            "*.txt",
            "LICENSE*",
            "README*",
            "merges.txt",
            "vocab.json",
        ],
    )
    print(path)


if __name__ == "__main__":
    main()
