#!/usr/bin/env python3
"""Measure end-to-end generation after the full pack and runtime are available."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from nhdf_edge.runtime.qwen3_loader import load_qwen3_moe, load_tokenizer


class TokenTimer(StoppingCriteria):
    """Record the host timestamp after every generated token."""

    def __init__(self) -> None:
        self.timestamps: list[float] = []

    def __call__(self, input_ids, scores, **kwargs):
        self.timestamps.append(time.perf_counter())
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack_dir")
    parser.add_argument("--prompt", default="Write a concise explanation of local implicit zero-set memory.")
    parser.add_argument("--system", default="You are a concise, accurate assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output")
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="research-only override needed while establishing pack quality",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.pack_dir)
    messages = [{"role": "system", "content": args.system}, {"role": "user", "content": args.prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = load_qwen3_moe(
        args.pack_dir,
        device="cuda",
        dtype=torch.float16,
        allow_unvalidated=args.allow_unvalidated,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    load_peak_allocated = torch.cuda.max_memory_allocated()
    load_peak_reserved = torch.cuda.max_memory_reserved()
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")

    # Warm-up catches integration and kernel errors before measured runs.
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=2, do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    runs = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        timer = TokenTimer()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([timer]),
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = int(output.shape[-1] - inputs["input_ids"].shape[-1])
        timestamps = timer.timestamps[:generated]
        ttft = timestamps[0] - start if timestamps else elapsed
        intervals = [timestamps[index] - timestamps[index - 1] for index in range(1, len(timestamps))]
        runs.append(
            {
                "seconds": elapsed,
                "generated_tokens": generated,
                "generated_text": tokenizer.decode(
                    output[0, inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                ),
                "time_to_first_token_seconds": ttft,
                "prefill_tokens_per_second": int(inputs["input_ids"].shape[-1]) / max(ttft, 1e-12),
                "steady_decode_tokens_per_second": (len(intervals) / sum(intervals)) if intervals else None,
                "median_inter_token_seconds": statistics.median(intervals) if intervals else None,
                "end_to_end_tokens_per_second": generated / elapsed,
            }
        )

    result = {
        "device": torch.cuda.get_device_name(0),
        "system_prompt": args.system,
        "user_prompt": args.prompt,
        "decoding": "greedy",
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "requested_new_tokens": args.max_new_tokens,
        "model_load_seconds": load_seconds,
        "model_load_peak_allocated_gb": load_peak_allocated / 1e9,
        "model_load_peak_reserved_gb": load_peak_reserved / 1e9,
        "runs": runs,
        "generation_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "generation_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "runtime_budget": model.nhdf_runtime_budget,
        "cache_implementation": model.generation_config.cache_implementation,
        "cache_config": model.generation_config.cache_config,
        "status": "measured-token-timed-generation",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
