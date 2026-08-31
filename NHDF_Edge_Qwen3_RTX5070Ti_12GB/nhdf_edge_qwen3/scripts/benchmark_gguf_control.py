#!/usr/bin/env python3
"""Run deterministic functional gates against an equal-budget GGUF control.

The prompt is rendered explicitly with the current non-thinking Qwen ChatML
shape.  This avoids treating an obsolete chat template embedded in an older
quantization as a model-quality failure.  Every prompt starts a fresh process,
and NVIDIA's device-wide used-memory counter is sampled while it runs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_RUNTIME = "tools/llama.cpp-b6014/bin/llama-cli.exe"
DEFAULT_MODEL = (
    "models/Qwen3-30B-A3B-Instruct-2507-IQ2_M/"
    "Qwen_Qwen3-30B-A3B-Instruct-2507-IQ2_M.gguf"
)
DEFAULT_OUTPUT = "metrics/local/gguf_iq2m_functional_gate.json"

PROMPTS = (
    {
        "id": "exact_ok",
        "user": "Reply with exactly the single word OK.",
        "max_tokens": 8,
        "accept": {"kind": "exact", "value": "OK"},
    },
    {
        "id": "arithmetic",
        "user": "Compute 17 multiplied by 19. Reply with only the integer.",
        "max_tokens": 12,
        "accept": {"kind": "regex", "value": r"^323\.?$"},
    },
    {
        "id": "integrity_vs_quality",
        "user": (
            "In one concise sentence, explain why a checksum passing does not prove "
            "that a compressed language model still produces useful answers."
        ),
        "max_tokens": 64,
        "accept": {
            "kind": "terms",
            "all_groups": [["checksum", "integrity"], ["quality", "accur", "semantic", "useful"]],
        },
    },
    {
        "id": "code",
        "user": (
            "Write only a short Python function named is_even(n) that returns whether "
            "an integer is even."
        ),
        "max_tokens": 64,
        "accept": {"kind": "terms", "all_groups": [["def is_even"], ["% 2", "& 1"]]},
    },
)


def _chatml(user: str) -> str:
    return (
        "<|im_start|>system\nYou are a precise assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _gpu_sample() -> tuple[int, int] | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        return None
    first = completed.stdout.strip().splitlines()[0]
    fields = [part.strip() for part in first.split(",")]
    return int(fields[0]), int(fields[1])


def _sample_until(stop: threading.Event, samples: list[dict[str, float | int]]) -> None:
    while not stop.is_set():
        sample = _gpu_sample()
        if sample is not None:
            used, utilization = sample
            samples.append(
                {
                    "monotonic_seconds": time.perf_counter(),
                    "used_memory_mib": used,
                    "utilization_percent": utilization,
                }
            )
        stop.wait(0.10)


def _clean_generation(stdout: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stdout)
    text = text.replace("[end of text]", "")
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return text.strip()


def _accepted(text: str, rule: dict[str, Any]) -> bool:
    normalized = text.strip()
    kind = rule["kind"]
    if kind == "exact":
        return normalized == rule["value"]
    if kind == "regex":
        return re.fullmatch(rule["value"], normalized, flags=re.IGNORECASE) is not None
    if kind == "terms":
        lowered = normalized.lower()
        return all(any(term.lower() in lowered for term in group) for group in rule["all_groups"])
    raise ValueError(f"unsupported acceptance rule: {kind}")


def _metric(pattern: str, stderr: str) -> float | None:
    match = re.search(pattern, stderr)
    return float(match.group(1)) if match else None


def _run_prompt(
    runtime: Path,
    model: Path,
    prompt: dict[str, Any],
    *,
    context: int,
    seed: int,
) -> dict[str, Any]:
    command = [
        str(runtime),
        "-m",
        str(model),
        "-ngl",
        "999",
        "-c",
        str(context),
        "-n",
        str(prompt["max_tokens"]),
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "-fa",
        "--temp",
        "0",
        "--top-k",
        "1",
        "-s",
        str(seed),
        "--no-display-prompt",
        "--simple-io",
        "--no-conversation",
        "--no-warmup",
        "-p",
        _chatml(prompt["user"]),
    ]
    baseline = _gpu_sample()
    samples: list[dict[str, float | int]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_until, args=(stop, samples), daemon=True)
    start = time.perf_counter()
    sampler.start()
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    elapsed = time.perf_counter() - start
    stop.set()
    sampler.join(timeout=2)
    generated = _clean_generation(process.stdout)
    used_samples = [int(sample["used_memory_mib"]) for sample in samples]
    utilization_samples = [int(sample["utilization_percent"]) for sample in samples]
    baseline_used = baseline[0] if baseline else None
    peak_used = max(used_samples) if used_samples else None
    return {
        "id": prompt["id"],
        "user": prompt["user"],
        "acceptance_rule": prompt["accept"],
        "passed": process.returncode == 0 and _accepted(generated, prompt["accept"]),
        "exit_code": process.returncode,
        "generated_text": generated,
        "elapsed_seconds": elapsed,
        "baseline_gpu_memory_mib": baseline_used,
        "peak_gpu_memory_mib": peak_used,
        "incremental_peak_gpu_memory_mib": (
            peak_used - baseline_used
            if peak_used is not None and baseline_used is not None
            else None
        ),
        "peak_gpu_utilization_percent": max(utilization_samples) if utilization_samples else None,
        "samples": len(samples),
        "llama_metrics": {
            "load_ms": _metric(r"load time\s*=\s*([0-9.]+) ms", process.stderr),
            "prompt_tokens_per_second": _metric(
                r"prompt eval time.*?([0-9.]+) tokens per second", process.stderr
            ),
            "decode_tokens_per_second": _metric(
                r"(?<!prompt )eval time.*?([0-9.]+) tokens per second", process.stderr
            ),
            "total_ms": _metric(r"total time\s*=\s*([0-9.]+) ms", process.stderr),
            "cuda_model_buffer_mib": _metric(
                r"CUDA0 model buffer size\s*=\s*([0-9.]+) MiB", process.stderr
            ),
            "cuda_kv_buffer_mib": _metric(
                r"CUDA0 KV buffer size\s*=\s*([0-9.]+) MiB", process.stderr
            ),
            "cuda_compute_buffer_mib": _metric(
                r"CUDA0 compute buffer size\s*=\s*([0-9.]+) MiB", process.stderr
            ),
        },
        "stderr_tail": process.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only", choices=[prompt["id"] for prompt in PROMPTS], action="append")
    args = parser.parse_args()

    runtime = Path(args.runtime).resolve()
    model = Path(args.model).resolve()
    if not runtime.is_file():
        parser.error(f"runtime not found: {runtime}")
    if not model.is_file():
        parser.error(f"model not found: {model}")
    selected = [prompt for prompt in PROMPTS if not args.only or prompt["id"] in args.only]

    results = [
        _run_prompt(runtime, model, prompt, context=args.context, seed=args.seed)
        for prompt in selected
    ]
    passed = all(result["passed"] for result in results)
    payload = {
        "experiment": "equal_budget_gguf_functional_control",
        "scope": "external control; not an NHDF codec validation",
        "runtime": str(runtime),
        "runtime_version": "llama.cpp b6014",
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "context_tokens": args.context,
        "kv_cache": "q8_0 K/V with Flash Attention",
        "gpu": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
        "results": results,
        "passed": passed,
        "status": "functional-control-pass" if passed else "functional-control-fail",
        "interpretation": (
            "A pass establishes that the exact model, GPU and byte budget can preserve useful "
            "behavior. It does not validate the failed NHDF scalar codec."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
