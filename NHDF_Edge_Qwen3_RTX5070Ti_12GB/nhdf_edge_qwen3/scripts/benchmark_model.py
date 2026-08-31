#!/usr/bin/env python3
"""Measure end-to-end generation after the full pack and runtime are available."""
from __future__ import annotations

import argparse
import json
import time

import torch

from nhdf_edge.runtime.qwen3_loader import load_qwen3_moe, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack_dir")
    parser.add_argument("--prompt", default="Write a concise explanation of local implicit zero-set memory.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.pack_dir)
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = load_qwen3_moe(args.pack_dir, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    # Warm-up catches integration and kernel errors before measured runs.
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=2, do_sample=False)
    torch.cuda.synchronize()

    runs = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = int(output.shape[-1] - inputs["input_ids"].shape[-1])
        runs.append({"seconds": elapsed, "generated_tokens": generated, "end_to_end_tps": generated / elapsed})

    result = {
        "device": torch.cuda.get_device_name(0),
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "requested_new_tokens": args.max_new_tokens,
        "model_load_seconds": load_seconds,
        "runs": runs,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "status": "measured-end-to-end; includes prefill and decode",
        "note": "Use a dedicated token-by-token harness for TTFT and steady-state decode separation.",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
