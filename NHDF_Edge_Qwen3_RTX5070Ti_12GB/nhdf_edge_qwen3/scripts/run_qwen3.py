#!/usr/bin/env python3
"""Minimal text-generation entry point for a completed NHDF pack."""
from __future__ import annotations

import argparse

import torch

from nhdf_edge.runtime.qwen3_loader import load_qwen3_moe, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack_dir")
    parser.add_argument("--prompt", default="Explain why local zero-set constraints are non-degenerate.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.pack_dir)
    model = load_qwen3_moe(args.pack_dir, device="cuda", dtype=torch.float16)
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
