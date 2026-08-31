#!/usr/bin/env python3
"""Download the selected Hugging Face checkpoint without loading it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--token")
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_revision = HfApi().model_info(
        args.repo,
        revision=args.revision,
        token=args.token,
    ).sha
    path = snapshot_download(
        repo_id=args.repo,
        # Pin the immutable commit resolved at the start of the download so a
        # moving branch cannot mix files from different upstream revisions.
        revision=resolved_revision,
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
    (target / "NHDF_SOURCE.json").write_text(
        json.dumps(
            {
                "repo_id": args.repo,
                "requested_revision": args.revision,
                "resolved_revision": resolved_revision,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
