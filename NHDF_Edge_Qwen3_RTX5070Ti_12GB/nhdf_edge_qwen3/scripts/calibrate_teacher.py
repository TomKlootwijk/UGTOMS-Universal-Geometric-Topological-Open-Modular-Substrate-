#!/usr/bin/env python3
"""Collect optional activation second moments from a teacher model.

This script is resource-intensive because the BF16 teacher does not fit in the
12 GB target. Run with CPU offload, on a larger machine, or skip it and use the
data-free packer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nhdf_edge.calibration import SecondMomentCollector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("prompts_jsonl", help="JSONL with a string field named 'text'")
    parser.add_argument("output")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install runtime dependencies: pip install -e '.[runtime]'") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    collector = SecondMomentCollector()
    collector.attach(model)
    count = 0
    with Path(args.prompts_jsonl).open("r", encoding="utf-8") as handle, torch.inference_mode():
        for line in handle:
            if count >= args.max_samples:
                break
            text = json.loads(line)["text"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            first_device = next(model.parameters()).device
            inputs = {key: value.to(first_device) for key, value in inputs.items()}
            model(**inputs, use_cache=False)
            count += 1
    collector.detach()
    collector.save(args.output, metadata={"samples": str(count), "model": str(args.model_dir)})
    print(args.output)


if __name__ == "__main__":
    main()
