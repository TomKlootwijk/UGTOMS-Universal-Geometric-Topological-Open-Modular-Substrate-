#!/usr/bin/env python3
"""Gate a GEMQ-style expert codec on real Qwen layer-0 activations.

This experiment deliberately does not load the complete 61 GB model.  It
memory-maps the source checkpoint, materializes only layer-0 attention/router
weights plus one expert at a time, and gathers expert inputs selected by the
real layer-0 router.  Calibration prompts and holdout prompts are disjoint.

The GPTQ update follows GEMQ's MIT-licensed implementation at commit
5eb2240cb46d9811bc9f79026100b46f62a7b642, adapted here to avoid a dependency
on HQQ/GemLite during this quality-only gate:
https://github.com/jndeng/GEMQ/blob/5eb2240cb46d9811bc9f79026100b46f62a7b642/gemq/quantizers/gptq.py

No packed checkpoint is written.  Reported storage bytes model GEMQ's runtime
layout: int32 column packing plus FP16 scale and zero for every 128 weights.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Must be set before CUDA creates a cuBLAS handle when deterministic algorithms
# are requested below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import transformers
from safetensors import safe_open
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeRotaryEmbedding,
)


GEMQ_COMMIT = "5eb2240cb46d9811bc9f79026100b46f62a7b642"
DEFAULT_SOURCE = "models/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_OUTPUT = "metrics/local/gemq_layer0_real_activation_gate.json"
DEFAULT_EXPERTS = (0, 17, 127)


CALIBRATION_PROMPTS = (
    "Explain why the sky appears blue during the day in two concise sentences.",
    "Write a Python function that returns all prime numbers below n and explain its complexity.",
    "Solve 17 multiplied by 23, then verify the result using a second method.",
    "Translate into Dutch: The quick brown fox jumps over the lazy dog.",
    "Compare TCP and UDP for an engineer choosing a protocol for real-time voice chat.",
    "A train travels 120 km in 90 minutes. Compute its average speed in km/h.",
    "Summarize the causes of ocean tides without using more than eighty words.",
    "Return valid JSON with keys name, version, and enabled for a fictional service.",
    "Explain the difference between correlation and causation with a concrete example.",
    "Draft a polite email declining a meeting because of a scheduling conflict.",
    "Find the next three terms in the sequence 2, 6, 12, 20, 30 and state the rule.",
    "Describe how a hash table handles collisions and give two common strategies.",
    "Rewrite this sentence in the passive voice: The committee approved the proposal.",
    "List five checks you would perform before deploying a database migration.",
    "Prove that the sum of two even integers is even using simple algebra.",
    "Create a short SQL query that counts orders per customer and sorts descending.",
    "A monitoring service receives timestamps, host names, latency values, and status codes from five regions. Design a compact incident-triage procedure that distinguishes a regional outage from a database bottleneck. Include the measurements you would compare, two plausible false alarms, and the evidence required before paging an engineer.",
    "Work through this planning problem carefully: a library has 240 metres of shelf space, fiction uses three eighths, history uses one fifth, and reference books use 36 metres. Calculate the unused space, show every intermediate quantity, and explain one independent check on the arithmetic.",
    "Review a hypothetical Python data pipeline that reads CSV rows, normalizes email addresses, joins customer records, and writes a report. Describe how you would make it deterministic, memory bounded, restartable after failure, and safe against malformed input. Give a small example for each property.",
    "Explain to a new systems programmer how virtual memory, physical memory, the page cache, and memory-mapped files interact when a process reads a model checkpoint much larger than RAM. Distinguish reserved address space from resident pages and identify two measurements that reveal actual pressure.",
    "A product team reports that conversion rose from 4.0 percent to 4.4 percent after a redesign. Describe the experiment needed to decide whether the redesign caused the change. Discuss randomization, sample size, confidence intervals, novelty effects, guardrail metrics, and what result would justify shipping.",
    "Write a precise algorithm for merging overlapping calendar intervals where endpoints may be inclusive, time zones may differ, and daylight-saving transitions are possible. State your normalization assumptions, give pseudocode, and walk through an example containing three partially overlapping intervals.",
    "Compare three ways to store a large immutable tensor collection: one monolithic binary file, many small files, and a sharded indexed format. Evaluate startup latency, random access, integrity verification, portability, and recovery from partial downloads, then recommend one for a laptop inference system.",
    "A sensor is specified as accurate within plus or minus 0.5 degrees, but repeated readings also contain random noise. Explain accuracy, precision, calibration bias, resolution, and uncertainty propagation. Then propose a short measurement protocol that produces an honest final temperature estimate.",
    "Create a database migration plan that splits a full-name column into given-name and family-name fields without downtime. Cover backward-compatible schema changes, dual writes, historical backfill, validation queries, gradual read switching, rollback, and removal of the old field.",
    "Analyze an API that accepts idempotency keys for payment requests. Explain exactly what the server should persist, how retries and concurrent duplicates behave, when records may expire, and how clients can distinguish a safe retry from a genuinely new transaction.",
    "Teach the difference between symmetric encryption, public-key encryption, hashing, and message authentication codes through one coherent file-transfer example. State which security property each primitive provides and call out one misuse that would leave the transfer vulnerable.",
    "A warehouse robot chooses among several routes with different lengths, congestion probabilities, and battery costs. Formulate a simple scoring rule, calculate it for three invented routes, explain the tradeoffs in the weights, and describe how observed outcomes should update future choices.",
    "Draft a code-review checklist for a concurrent queue implementation. Include invariants, lock ordering, lost wakeups, cancellation, memory visibility, shutdown behavior, stress testing, and performance counters. For three items, give a concrete failure scenario the reviewer should try to reproduce.",
    "Summarize how DNS resolution proceeds from an application cache through recursive and authoritative servers. Include positive and negative caching, TTL behavior, CNAME chains, DNSSEC at a high level, and why changing a record does not become visible everywhere immediately.",
    "Design a reproducible benchmark comparing two compression codecs for neural-network weights. Specify immutable inputs, calibration and holdout separation, storage accounting, numerical metrics, warmup, peak-memory measurement, failure thresholds, and which conclusions the benchmark cannot support.",
    "An organization wants to rotate an authentication signing key without logging out every user. Propose a staged procedure involving key identifiers, overlapping verification windows, cache refresh, monitoring, emergency rollback, and eventual retirement. Explain how to handle tokens issued just before each transition.",
)


HOLDOUT_PROMPTS = (
    "What is 91 divided by 7? Answer with the number and one verification step.",
    "Explain photosynthesis to a twelve-year-old using exactly three sentences.",
    "Implement binary search in pseudocode and identify its worst-case complexity.",
    "Translate into French: Reliable measurements matter more than optimistic claims.",
    "Give two arguments for and two arguments against remote work for software teams.",
    "A box has dimensions 3, 4, and 5 metres. Calculate its volume and surface area.",
    "Correct the grammar in this sentence: Neither of the reports were finished on time.",
    "Describe a safe rollback plan for a failed production release in four steps.",
    "Why does adding salt lower the freezing point of water? Keep the answer concise.",
    "Write a regular expression that matches a simple YYYY-MM-DD date and note its limits.",
    "Compute the greatest common divisor of 84 and 126 and show the Euclidean steps.",
    "State the difference between authentication and authorization with one example each.",
    "A web cache serves stale data after a deployment. List the observations needed to distinguish browser caching, CDN caching, application caching, and a stale database replica, then propose the safest order in which to test those hypotheses.",
    "Calculate the compound value of 1200 euros invested for three years at five percent annual interest. Show the formula, the value after each year, and explain why simple interest would give a different answer.",
    "Describe how to parse an untrusted length-prefixed binary message safely. Address integer overflow, maximum sizes, truncated input, unknown fields, allocation limits, and how the caller should receive structured errors.",
    "Explain why a randomized controlled trial can still produce a misleading result. Discuss attrition, multiple comparisons, noncompliance, measurement error, and limited external validity using a concise example.",
    "Write pseudocode for a rate limiter using a token bucket. Define capacity, refill, burst behavior, clock handling, concurrency requirements, and the response returned when a request is rejected.",
    "A backup job reports success but restoration fails. Produce a verification plan that covers checksums, catalog consistency, encryption keys, dependency ordering, point-in-time recovery, and a scheduled restore drill.",
    "Compare breadth-first search and depth-first search on a finite graph. State their memory and time costs, identify a task suited to each, and demonstrate the visitation order on a small graph you define.",
    "Explain floating-point cancellation with a numerical example, then give two implementation techniques that reduce its effect and one test that would catch the resulting loss of precision.",
    "Design a safe feature-flag rollout for a new request-routing algorithm. Include cohort selection, shadow evaluation, progressive percentages, automatic rollback criteria, observability, and cleanup after the rollout succeeds.",
    "A service-level objective allows 43 minutes of downtime in a thirty-day month. Explain how an error budget is consumed, what burn rate means, and how alert thresholds could detect both fast and slow incidents.",
    "Provide a normalized relational schema for authors, books, editions, and publishers. State the primary and foreign keys, handle a book with multiple authors, and give one query that lists every edition for an author.",
    "Explain the difference between a cryptographic checksum and an ordinary error-detecting checksum. Recommend one for verifying a downloaded model shard and justify the choice in terms of accidental corruption and malicious replacement.",
)


def _parse_experts(value: str) -> tuple[int, ...]:
    experts = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    if not experts or any(expert < 0 or expert >= 128 for expert in experts):
        raise argparse.ArgumentTypeError("experts must be comma-separated integers in [0, 127]")
    return experts


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(hidden.dtype) * weight


def _comparison(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.float()
    reference = reference.float()
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise RuntimeError("non-finite value encountered in expert-output comparison")
    diff = actual - reference
    rmse = diff.square().mean().sqrt()
    reference_rms = reference.square().mean().sqrt()
    return {
        "rmse": float(rmse),
        "reference_rms": float(reference_rms),
        "normalized_rmse": float(rmse / reference_rms.clamp_min(1e-12)),
        "max_abs": float(diff.abs().max()),
        "cosine_similarity": float(
            F.cosine_similarity(actual.reshape(1, -1), reference.reshape(1, -1), dim=-1)
        ),
    }


def _prompt_fingerprint(prompts: Iterable[str]) -> str:
    payload = json.dumps(list(prompts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _peak_host_working_set_bytes() -> int | None:
    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    except (ImportError, OSError):
        return None


class CheckpointReader:
    """Small safetensors reader that never assembles the complete checkpoint."""

    def __init__(self, source: Path) -> None:
        self.source = source
        index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]

    def tensor(self, name: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        with safe_open(
            str(self.source / self.weight_map[name]), framework="pt", device="cpu"
        ) as handle:
            value = handle.get_tensor(name)
        return value.to(device=device, dtype=dtype)

    def embedding_handle(self):
        name = "model.embed_tokens.weight"
        return safe_open(
            str(self.source / self.weight_map[name]), framework="pt", device="cpu"
        )


def _load_layer0_front(
    reader: CheckpointReader,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Qwen3MoeAttention, Qwen3MoeRotaryEmbedding, torch.Tensor, torch.Tensor, torch.Tensor]:
    config._attn_implementation = "eager"
    attention = Qwen3MoeAttention(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        for name, parameter in attention.named_parameters():
            source_name = f"model.layers.0.self_attn.{name}"
            parameter.copy_(reader.tensor(source_name, device=device, dtype=dtype))

    rotary = Qwen3MoeRotaryEmbedding(config, device=device)
    input_norm = reader.tensor(
        "model.layers.0.input_layernorm.weight", device=device, dtype=dtype
    )
    post_attention_norm = reader.tensor(
        "model.layers.0.post_attention_layernorm.weight", device=device, dtype=dtype
    )
    router = reader.tensor("model.layers.0.mlp.gate.weight", device=device, dtype=dtype)
    return attention, rotary, input_norm, post_attention_norm, router


def _input_ids(encoded: Any) -> torch.Tensor:
    token_ids = encoded["input_ids"] if "input_ids" in encoded else encoded
    return token_ids[0] if token_ids.ndim == 2 else token_ids


def _prompt_token_ids(
    tokenizer: Any, prompt: str, max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    scenario_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise and helpful assistant. "
                f"Treat scenario {scenario_id} independently."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Transformers 5 returns a BatchEncoding here; indexing it numerically
    # yields a tokenizers.Encoding rather than the requested torch tensor.
    token_ids = _input_ids(encoded)
    content_ids = _input_ids(
        tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    )
    # Only score/gather the user-content span. The chat headers and shared
    # system message are still part of the causal context, but excluding their
    # rows prevents identical prefix activations leaking into both splits.
    token_list = token_ids.tolist()
    content_list = content_ids.tolist()
    candidates = [
        start
        for start in range(len(token_list) - len(content_list) + 1)
        if token_list[start : start + len(content_list)] == content_list
    ]
    if not candidates:
        raise RuntimeError("could not locate the user-content token span in the chat template")
    content_start = candidates[-1]
    eligible = torch.zeros(token_ids.shape[0], dtype=torch.bool)
    eligible[content_start : content_start + content_ids.numel()] = True
    if token_ids.numel() > max_tokens:
        token_ids = token_ids[-max_tokens:]
        eligible = eligible[-max_tokens:]
    return (
        token_ids.to(dtype=torch.long, device="cpu"),
        eligible.to(device="cpu"),
    )


def _embedding_lookup(handle: Any, token_ids: torch.Tensor, cache: dict[int, torch.Tensor]) -> torch.Tensor:
    name = "model.embed_tokens.weight"
    rows = []
    for token_id in token_ids.tolist():
        if token_id not in cache:
            cache[token_id] = handle.get_slice(name)[token_id : token_id + 1].clone()
        rows.append(cache[token_id])
    return torch.cat(rows, dim=0)


@torch.inference_mode()
def _capture_split(
    prompts: Iterable[str],
    *,
    tokenizer: Any,
    embedding_handle: Any,
    embedding_cache: dict[int, torch.Tensor],
    attention: Qwen3MoeAttention,
    rotary: Qwen3MoeRotaryEmbedding,
    input_norm_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    router_weight: torch.Tensor,
    experts: tuple[int, ...],
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    max_prompt_tokens: int,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    by_expert: dict[int, list[torch.Tensor]] = defaultdict(list)
    route_counts = torch.zeros(config.num_experts, dtype=torch.int64)
    prompt_records = []
    total_tokens = 0
    eligible_tokens = 0

    for prompt_index, prompt in enumerate(prompts):
        token_ids, eligible = _prompt_token_ids(tokenizer, prompt, max_prompt_tokens)
        embeddings = _embedding_lookup(embedding_handle, token_ids, embedding_cache)
        hidden = embeddings.to(device=device, dtype=dtype).unsqueeze(0)
        sequence_length = hidden.shape[1]
        total_tokens += sequence_length
        eligible = eligible.to(device=device)
        eligible_tokens += int(eligible.sum())

        position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
        normalized = _rms_norm(hidden, input_norm_weight, config.rms_norm_eps)
        position_embeddings = rotary(normalized, position_ids)
        causal = torch.triu(
            torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool),
            diagonal=1,
        )
        attention_mask = torch.zeros(
            (1, 1, sequence_length, sequence_length), device=device, dtype=dtype
        )
        attention_mask.masked_fill_(causal, torch.finfo(dtype).min)
        attention_output, _ = attention(
            normalized,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        expert_input = _rms_norm(
            hidden + attention_output, post_attention_norm_weight, config.rms_norm_eps
        )[0]

        router_logits = F.linear(expert_input, router_weight)
        selected = torch.topk(router_logits.float().softmax(dim=-1), config.num_experts_per_tok, dim=-1).indices
        eligible_selected = selected[eligible]
        route_counts += torch.bincount(
            eligible_selected.reshape(-1).cpu(), minlength=config.num_experts
        )
        prompt_hits = {}
        for expert in experts:
            mask = torch.any(selected == expert, dim=-1) & eligible
            prompt_hits[str(expert)] = int(mask.sum())
            if torch.any(mask):
                by_expert[expert].append(expert_input[mask].to(device="cpu", dtype=torch.float32))
        prompt_records.append(
            {
                "prompt_index": prompt_index,
                "tokens": sequence_length,
                "eligible_user_content_tokens": int(eligible.sum()),
                "target_expert_hits": prompt_hits,
            }
        )

    empty = torch.empty((0, config.hidden_size), dtype=torch.float32)
    combined = {
        expert: torch.cat(by_expert[expert], dim=0) if by_expert[expert] else empty.clone()
        for expert in experts
    }
    stats = {
        "prompts": len(prompt_records),
        "tokens": total_tokens,
        "eligible_user_content_tokens": eligible_tokens,
        "target_expert_hits": {str(expert): int(combined[expert].shape[0]) for expert in experts},
        "all_expert_route_counts": route_counts.tolist(),
        "prompt_records": prompt_records,
    }
    return combined, stats


def _find_params(
    values: torch.Tensor,
    nbits: int,
    *,
    mse: bool,
    mse_grid: int,
    max_shrink: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_int = 2**nbits - 1
    if max_int == 1:
        scales = values.abs().mean(dim=1, keepdim=True).clamp_min(1e-8) * 2
        zeros = torch.full_like(scales, 0.5)
        return scales, zeros

    zero = torch.zeros((values.shape[0], 1), device=values.device, dtype=values.dtype)
    minimum = torch.minimum(values.amin(dim=1, keepdim=True), zero)
    maximum = torch.maximum(values.amax(dim=1, keepdim=True), zero)
    all_zero = (minimum == 0) & (maximum == 0)
    minimum = torch.where(all_zero, torch.full_like(minimum, -1), minimum)
    maximum = torch.where(all_zero, torch.ones_like(maximum), maximum)
    scales = (maximum - minimum).clamp_min(1e-5) / max_int
    zeros = torch.round(-minimum / scales)

    if not mse:
        return scales, zeros

    best = torch.full((values.shape[0],), float("inf"), device=values.device)
    for index in range(int(max_shrink * mse_grid)):
        fraction = 1.0 - index / mse_grid
        candidate_minimum = fraction * minimum
        candidate_maximum = fraction * maximum
        candidate_scales = (candidate_maximum - candidate_minimum).clamp_min(1e-5) / max_int
        candidate_zeros = torch.round(-candidate_minimum / candidate_scales)
        codes = torch.clamp(torch.round(values / candidate_scales) + candidate_zeros, 0, max_int)
        reconstructed = (codes - candidate_zeros) * candidate_scales
        error = (reconstructed - values).abs().pow(2.4).sum(dim=1)
        improved = error < best
        if torch.any(improved):
            best[improved] = error[improved]
            scales[improved] = candidate_scales[improved]
            zeros[improved] = candidate_zeros[improved]
    return scales, zeros


def _quantize_vector(
    values: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, nbits: int
) -> torch.Tensor:
    if nbits == 1:
        codes = torch.where(values >= 0, 1.0, 0.0)
    else:
        codes = torch.clamp(torch.round(values / scales) + zeros, 0, 2**nbits - 1)
    return (codes - zeros) * scales


def _input_hessian(
    inputs: torch.Tensor, percdamp: float
) -> tuple[torch.Tensor, float, float]:
    values = inputs.float()
    hessian = 2.0 / max(values.shape[0], 1) * values.T.matmul(values)
    dead = torch.diag(hessian) == 0
    dead_fraction = float(dead.float().mean())
    hessian[dead, dead] = 1
    damp = float(percdamp * torch.diag(hessian).mean())
    diagonal = torch.arange(hessian.shape[0], device=hessian.device)
    hessian[diagonal, diagonal] += damp
    try:
        factor = torch.linalg.cholesky(hessian)
    except torch.linalg.LinAlgError as error:
        raise RuntimeError(
            f"GPTQ Hessian is not positive definite after dampening ({damp=:.6g})"
        ) from error
    inverse = torch.cholesky_inverse(factor)
    inverse_factor = torch.linalg.cholesky(inverse, upper=True)
    return inverse_factor, damp, dead_fraction


@torch.no_grad()
def _gptq_quantize(
    weight: torch.Tensor,
    inverse_hessian_factor: torch.Tensor,
    *,
    nbits: int,
    group_size: int,
    block_size: int,
    mse: bool,
    mse_grid: int,
) -> torch.Tensor:
    """Return dequantized GPTQ weights while retaining only bounded workspaces."""

    columns = weight.shape[1]
    if columns % group_size:
        raise ValueError(f"input width {columns} is not divisible by group size {group_size}")
    work = weight.float().clone()
    reconstructed = torch.zeros_like(work)

    for block_start in range(0, columns, block_size):
        block_end = min(block_start + block_size, columns)
        block_width = block_end - block_start
        work_block = work[:, block_start:block_end].clone()
        reconstructed_block = torch.zeros_like(work_block)
        errors = torch.zeros_like(work_block)
        inverse_block = inverse_hessian_factor[block_start:block_end, block_start:block_end]
        scales = zeros = None

        for local_column in range(block_width):
            global_column = block_start + local_column
            if global_column % group_size == 0:
                group = work[:, global_column : global_column + group_size]
                scales, zeros = _find_params(
                    group,
                    nbits,
                    mse=mse,
                    mse_grid=mse_grid,
                    max_shrink=0.8,
                )
            assert scales is not None and zeros is not None
            column = work_block[:, local_column]
            diagonal = inverse_block[local_column, local_column]
            quantized = _quantize_vector(column[:, None], scales, zeros, nbits).flatten()
            reconstructed_block[:, local_column] = quantized
            error = (column - quantized) / diagonal
            work_block[:, local_column:] -= error[:, None].matmul(
                inverse_block[local_column, local_column:][None, :]
            )
            errors[:, local_column] = error

        reconstructed[:, block_start:block_end] = reconstructed_block
        work[:, block_end:] -= errors.matmul(
            inverse_hessian_factor[block_start:block_end, block_end:]
        )
    return reconstructed


@torch.no_grad()
def _rtn_quantize(
    weight: torch.Tensor,
    *,
    nbits: int,
    group_size: int,
    mse: bool,
    mse_grid: int,
) -> torch.Tensor:
    rows, columns = weight.shape
    if columns % group_size:
        raise ValueError(f"input width {columns} is not divisible by group size {group_size}")
    groups = weight.float().reshape(rows * (columns // group_size), group_size)
    scales, zeros = _find_params(
        groups,
        nbits,
        mse=mse,
        mse_grid=mse_grid,
        max_shrink=0.8,
    )
    return _quantize_vector(groups, scales, zeros, nbits).reshape_as(weight)


def _runtime_bytes(shape: tuple[int, int], nbits: int, group_size: int) -> int:
    output_features, input_features = shape
    values_per_word = 32 // nbits
    packed = output_features * math.ceil(input_features / values_per_word) * 4
    groups = output_features * (input_features // group_size)
    scale_and_zero = groups * 2 * 2
    return packed + scale_and_zero


def _expert_output(
    inputs: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_projection = F.linear(inputs, gate)
    up_projection = F.linear(inputs, up)
    activated = F.silu(gate_projection) * up_projection
    output = F.linear(activated, down)
    return torch.cat((gate_projection, up_projection), dim=-1), activated, output


def _write_result(path: Path, result: dict[str, Any]) -> None:
    text = json.dumps(result, indent=2)
    print(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--experts", type=_parse_experts, default=DEFAULT_EXPERTS)
    parser.add_argument("--bits", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--mse-grid", type=int, default=100)
    parser.add_argument("--no-mse", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=160)
    parser.add_argument("--max-hits", type=int, default=256)
    parser.add_argument("--min-hits", type=int, default=8)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.group_size <= 0 or args.block_size <= 0 or args.mse_grid <= 0:
        parser.error("group size, block size, and mse grid must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this bounded gate requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    reader = CheckpointReader(source)
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    provenance_path = source / "NHDF_SOURCE.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else {}
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    attention, rotary, input_norm, post_norm, router = _load_layer0_front(
        reader, config, device, dtype
    )
    embedding_cache: dict[int, torch.Tensor] = {}
    with reader.embedding_handle() as embedding_handle:
        calibration, calibration_stats = _capture_split(
            CALIBRATION_PROMPTS,
            tokenizer=tokenizer,
            embedding_handle=embedding_handle,
            embedding_cache=embedding_cache,
            attention=attention,
            rotary=rotary,
            input_norm_weight=input_norm,
            post_attention_norm_weight=post_norm,
            router_weight=router,
            experts=args.experts,
            config=config,
            device=device,
            dtype=dtype,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        holdout, holdout_stats = _capture_split(
            HOLDOUT_PROMPTS,
            tokenizer=tokenizer,
            embedding_handle=embedding_handle,
            embedding_cache=embedding_cache,
            attention=attention,
            rotary=rotary,
            input_norm_weight=input_norm,
            post_attention_norm_weight=post_norm,
            router_weight=router,
            experts=args.experts,
            config=config,
            device=device,
            dtype=dtype,
            max_prompt_tokens=args.max_prompt_tokens,
        )

    capture_seconds = time.perf_counter() - start_time
    del attention, rotary, input_norm, post_norm, router, embedding_cache
    gc.collect()
    torch.cuda.empty_cache()

    base_result: dict[str, Any] = {
        "experiment": "gemq_style_gptq_real_activation_expert_gate",
        "scope": "isolated layer-0 expert outputs; not a full-model quality certificate",
        "source_model": str(source),
        "source_revision": provenance.get("resolved_revision"),
        "source_architecture": getattr(config, "architectures", None),
        "gemq_source_commit": GEMQ_COMMIT,
        "reproducibility": {
            "seed": args.seed,
            "deterministic_algorithms": True,
            "activation_sampling": (
                "real layer-0 routes at user-content token positions only; shared chat "
                "headers are excluded, and every prompt has a deterministic unique "
                "system-context fingerprint to prevent identical causal prefix rows"
            ),
            "calibration_prompt_sha256": _prompt_fingerprint(CALIBRATION_PROMPTS),
            "holdout_prompt_sha256": _prompt_fingerprint(HOLDOUT_PROMPTS),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "platform": sys.platform,
            "layer0_shard": reader.weight_map["model.embed_tokens.weight"],
            "layer0_shard_bytes": (
                source / reader.weight_map["model.embed_tokens.weight"]
            ).stat().st_size,
        },
        "layer": 0,
        "experts": list(args.experts),
        "quantization": {
            "bits": args.bits,
            "group_size": args.group_size,
            "block_size": args.block_size,
            "percdamp": args.percdamp,
            "mse_range_search": not args.no_mse,
            "mse_grid": args.mse_grid,
            "packing_model": "GemLite int32 columns + FP16 scale and zero",
        },
        "calibration_capture": calibration_stats,
        "holdout_capture": holdout_stats,
        "capture_seconds": capture_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_host_working_set_bytes": _peak_host_working_set_bytes(),
    }

    insufficient = {
        expert: {
            "calibration": int(calibration[expert].shape[0]),
            "holdout": int(holdout[expert].shape[0]),
        }
        for expert in args.experts
        if calibration[expert].shape[0] < args.min_hits or holdout[expert].shape[0] < args.min_hits
    }
    if insufficient:
        base_result.update(
            {
                "status": "rejected-insufficient-real-router-hits",
                "insufficient": insufficient,
            }
        )
        _write_result(output, base_result)
        raise SystemExit(2)
    if args.capture_only:
        base_result["status"] = "capture-only"
        _write_result(output, base_result)
        return

    for expert in args.experts:
        calibration[expert] = calibration[expert][: args.max_hits].contiguous()
        holdout[expert] = holdout[expert][: args.max_hits].contiguous()

    expert_results: dict[str, Any] = {}
    quantization_start = time.perf_counter()
    for expert in args.experts:
        prefix = f"model.layers.0.mlp.experts.{expert}"
        gate = reader.tensor(f"{prefix}.gate_proj.weight", device=device, dtype=torch.float32)
        up = reader.tensor(f"{prefix}.up_proj.weight", device=device, dtype=torch.float32)
        down = reader.tensor(f"{prefix}.down_proj.weight", device=device, dtype=torch.float32)
        calibration_input = calibration[expert].to(device=device)
        holdout_input = holdout[expert].to(device=device)

        calibration_hidden = F.silu(F.linear(calibration_input, gate)) * F.linear(
            calibration_input, up
        )
        input_inverse, input_damp, input_dead_fraction = _input_hessian(
            calibration_input, args.percdamp
        )
        down_inverse, down_damp, down_dead_fraction = _input_hessian(
            calibration_hidden, args.percdamp
        )

        expert_start = time.perf_counter()
        gptq_gate = _gptq_quantize(
            gate,
            input_inverse,
            nbits=args.bits,
            group_size=args.group_size,
            block_size=args.block_size,
            mse=not args.no_mse,
            mse_grid=args.mse_grid,
        )
        gptq_up = _gptq_quantize(
            up,
            input_inverse,
            nbits=args.bits,
            group_size=args.group_size,
            block_size=args.block_size,
            mse=not args.no_mse,
            mse_grid=args.mse_grid,
        )
        gptq_down = _gptq_quantize(
            down,
            down_inverse,
            nbits=args.bits,
            group_size=args.group_size,
            block_size=args.block_size,
            mse=not args.no_mse,
            mse_grid=args.mse_grid,
        )
        rtn_gate = _rtn_quantize(
            gate,
            nbits=args.bits,
            group_size=args.group_size,
            mse=not args.no_mse,
            mse_grid=args.mse_grid,
        )
        rtn_up = _rtn_quantize(
            up,
            nbits=args.bits,
            group_size=args.group_size,
            mse=not args.no_mse,
            mse_grid=args.mse_grid,
        )
        rtn_down = _rtn_quantize(
            down,
            nbits=args.bits,
            group_size=args.group_size,
            mse=not args.no_mse,
            mse_grid=args.mse_grid,
        )
        torch.cuda.synchronize(device)
        expert_seconds = time.perf_counter() - expert_start

        reference_projection, reference_hidden, reference_output = _expert_output(
            holdout_input, gate, up, down
        )
        gptq_projection, gptq_hidden, gptq_output = _expert_output(
            holdout_input, gptq_gate, gptq_up, gptq_down
        )
        rtn_projection, rtn_hidden, rtn_output = _expert_output(
            holdout_input, rtn_gate, rtn_up, rtn_down
        )

        shapes = (tuple(gate.shape), tuple(up.shape), tuple(down.shape))
        packed_bytes = sum(_runtime_bytes(shape, args.bits, args.group_size) for shape in shapes)
        parameters = sum(math.prod(shape) for shape in shapes)
        gptq_metrics = {
            "gate_up_projection": _comparison(gptq_projection, reference_projection),
            "activated_hidden": _comparison(gptq_hidden, reference_hidden),
            "expert_output": _comparison(gptq_output, reference_output),
        }
        rtn_metrics = {
            "gate_up_projection": _comparison(rtn_projection, reference_projection),
            "activated_hidden": _comparison(rtn_hidden, reference_hidden),
            "expert_output": _comparison(rtn_output, reference_output),
        }
        baseline_nrmse = rtn_metrics["expert_output"]["normalized_rmse"]
        candidate_nrmse = gptq_metrics["expert_output"]["normalized_rmse"]
        expert_results[str(expert)] = {
            "calibration_hits_used": int(calibration_input.shape[0]),
            "holdout_hits_used": int(holdout_input.shape[0]),
            "input_hessian_damp": input_damp,
            "down_hessian_damp": down_damp,
            "input_hessian_dead_diagonal_fraction": input_dead_fraction,
            "down_hessian_dead_diagonal_fraction": down_dead_fraction,
            "runtime_packed_bytes": packed_bytes,
            "effective_bits_per_weight": packed_bytes * 8.0 / parameters,
            "quantization_seconds": expert_seconds,
            "gptq": gptq_metrics,
            "equal_storage_rtn": rtn_metrics,
            "relative_nrmse_improvement_vs_rtn": (
                baseline_nrmse - candidate_nrmse
            )
            / max(baseline_nrmse, 1e-12),
        }

        del (
            gate,
            up,
            down,
            calibration_input,
            holdout_input,
            calibration_hidden,
            input_inverse,
            down_inverse,
            gptq_gate,
            gptq_up,
            gptq_down,
            rtn_gate,
            rtn_up,
            rtn_down,
            reference_projection,
            reference_hidden,
            reference_output,
            gptq_projection,
            gptq_hidden,
            gptq_output,
            rtn_projection,
            rtn_hidden,
            rtn_output,
        )
        gc.collect()
        torch.cuda.empty_cache()

    output_nrmse = [
        expert_results[str(expert)]["gptq"]["expert_output"]["normalized_rmse"]
        for expert in args.experts
    ]
    output_cosine = [
        expert_results[str(expert)]["gptq"]["expert_output"]["cosine_similarity"]
        for expert in args.experts
    ]
    improvements = [
        expert_results[str(expert)]["relative_nrmse_improvement_vs_rtn"]
        for expert in args.experts
    ]
    thresholds = {
        "worst_expert_output_nrmse_max": 0.35,
        "worst_expert_output_cosine_min": 0.94,
        "median_expert_output_nrmse_max": 0.30,
        "median_expert_output_cosine_min": 0.95,
        "minimum_relative_nrmse_improvement_vs_equal_storage_rtn": 0.20,
    }
    aggregate = {
        "worst_expert_output_nrmse": max(output_nrmse),
        "worst_expert_output_cosine": min(output_cosine),
        "median_expert_output_nrmse": statistics.median(output_nrmse),
        "median_expert_output_cosine": statistics.median(output_cosine),
        "minimum_relative_nrmse_improvement_vs_equal_storage_rtn": min(improvements),
    }
    passed = (
        aggregate["worst_expert_output_nrmse"] <= thresholds["worst_expert_output_nrmse_max"]
        and aggregate["worst_expert_output_cosine"] >= thresholds["worst_expert_output_cosine_min"]
        and aggregate["median_expert_output_nrmse"] <= thresholds["median_expert_output_nrmse_max"]
        and aggregate["median_expert_output_cosine"] >= thresholds["median_expert_output_cosine_min"]
        and aggregate["minimum_relative_nrmse_improvement_vs_equal_storage_rtn"]
        >= thresholds["minimum_relative_nrmse_improvement_vs_equal_storage_rtn"]
    )
    base_result.update(
        {
            "expert_results": expert_results,
            "gate_thresholds": thresholds,
            "aggregate": aggregate,
            "quantization_seconds": time.perf_counter() - quantization_start,
            "total_seconds": time.perf_counter() - start_time,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_host_working_set_bytes": _peak_host_working_set_bytes(),
            "status": "pass-isolated-expert-gate" if passed else "reject-isolated-expert-gate",
            "next_gate": (
                "router-weighted full layer-0 output on disjoint prompts"
                if passed
                else "do not repack; codec failed the declared isolated-expert threshold"
            ),
        }
    )
    _write_result(output, base_result)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
