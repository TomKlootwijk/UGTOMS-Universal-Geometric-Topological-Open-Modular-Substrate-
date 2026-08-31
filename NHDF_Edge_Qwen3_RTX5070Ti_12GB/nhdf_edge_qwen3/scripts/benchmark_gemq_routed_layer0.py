#!/usr/bin/env python3
"""Benchmark a complete router-weighted Qwen3-MoE layer-0 expert output.

This is the bounded follow-up to ``benchmark_gemq_real_activation.py``.  It
uses the same disjoint calibration and holdout prompts, runs the real BF16
layer-0 attention and router, and then evaluates every expert selected by the
holdout top-8 routes.  Experts are loaded and quantized one at a time, so the
complete 30B checkpoint and the complete layer are never resident in RAM or
VRAM.

The candidate is the proposed mixed representation for this sublayer: the
router remains source BF16, sufficiently observed experts use GEMQ-style GPTQ,
and experts without enough real calibration routes fall back to equal-storage
RTN.  The control keeps the identical BF16 router and uses MSE-optimized RTN at
exactly the same physical expert layout.  Both are compared with the source
BF16-valued expert weights at the complete weighted MoE output and after the
decoder residual addition.

The GPTQ implementation, packing model, prompt corpus, and checkpoint reader
are imported from the isolated-expert gate rather than silently creating a
second quantizer implementation.
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
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import transformers
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer

import benchmark_gemq_real_activation as expert_gate


DEFAULT_OUTPUT = "metrics/local/gemq_router_weighted_layer0_gate.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def _capture_split(
    prompts: Iterable[str],
    *,
    tokenizer: Any,
    embedding_handle: Any,
    embedding_cache: dict[int, torch.Tensor],
    attention: torch.nn.Module,
    rotary: torch.nn.Module,
    input_norm_weight: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    router_weight: torch.Tensor,
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
    max_prompt_tokens: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Capture all eligible token rows plus their exact source top-k routes."""

    expert_inputs: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    route_indices: list[torch.Tensor] = []
    route_weights: list[torch.Tensor] = []
    route_counts = torch.zeros(config.num_experts, dtype=torch.int64)
    prompt_records: list[dict[str, Any]] = []
    total_tokens = 0
    eligible_tokens = 0

    for prompt_index, prompt in enumerate(prompts):
        token_ids, eligible = expert_gate._prompt_token_ids(
            tokenizer, prompt, max_prompt_tokens
        )
        embeddings = expert_gate._embedding_lookup(
            embedding_handle, token_ids, embedding_cache
        )
        hidden = embeddings.to(device=device, dtype=dtype).unsqueeze(0)
        sequence_length = hidden.shape[1]
        total_tokens += sequence_length
        eligible = eligible.to(device=device)
        prompt_eligible = int(eligible.sum())
        eligible_tokens += prompt_eligible

        position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
        normalized = expert_gate._rms_norm(
            hidden, input_norm_weight, config.rms_norm_eps
        )
        position_embeddings = rotary(normalized, position_ids)
        causal = torch.triu(
            torch.ones(
                (sequence_length, sequence_length),
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        attention_mask = torch.zeros(
            (1, 1, sequence_length, sequence_length),
            device=device,
            dtype=dtype,
        )
        attention_mask.masked_fill_(causal, torch.finfo(dtype).min)
        attention_output, _ = attention(
            normalized,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        residual = (hidden + attention_output)[0]
        expert_input = expert_gate._rms_norm(
            residual, post_attention_norm_weight, config.rms_norm_eps
        )

        router_logits = F.linear(expert_input, router_weight)
        router_probabilities = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        weights, indices = torch.topk(
            router_probabilities, config.num_experts_per_tok, dim=-1
        )
        if config.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        # Match Qwen3MoeTopKRouter: normalized weights are cast back to the
        # router-logit dtype before expert accumulation.
        weights = weights.to(router_logits.dtype)

        selected_indices = indices[eligible]
        selected_weights = weights[eligible]
        route_counts += torch.bincount(
            selected_indices.reshape(-1).cpu(), minlength=config.num_experts
        )
        start = sum(value.shape[0] for value in expert_inputs)
        end = start + prompt_eligible
        expert_inputs.append(
            expert_input[eligible].to(device="cpu", dtype=torch.float32)
        )
        residuals.append(residual[eligible].to(device="cpu", dtype=torch.float32))
        route_indices.append(selected_indices.to(device="cpu", dtype=torch.int64))
        route_weights.append(
            selected_weights.to(device="cpu", dtype=torch.float32)
        )
        prompt_records.append(
            {
                "prompt_index": prompt_index,
                "tokens": sequence_length,
                "eligible_user_content_tokens": prompt_eligible,
                "captured_start": start,
                "captured_end": end,
            }
        )

    captured = {
        "expert_input": torch.cat(expert_inputs, dim=0),
        "residual": torch.cat(residuals, dim=0),
        "route_indices": torch.cat(route_indices, dim=0),
        "route_weights": torch.cat(route_weights, dim=0),
    }
    stats = {
        "prompts": len(prompt_records),
        "tokens": total_tokens,
        "eligible_user_content_tokens": eligible_tokens,
        "top_k": config.num_experts_per_tok,
        "all_expert_route_counts": route_counts.tolist(),
        "experts_with_routes": int((route_counts > 0).sum()),
        "prompt_records": prompt_records,
    }
    return captured, stats


def _sample_calibration_rows(
    values: torch.Tensor, maximum: int, seed: int
) -> tuple[torch.Tensor, list[int]]:
    if maximum <= 0 or values.shape[0] <= maximum:
        indices = torch.arange(values.shape[0], dtype=torch.int64)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        indices = torch.randperm(values.shape[0], generator=generator)[:maximum]
        indices = torch.sort(indices).values
    return values[indices].contiguous(), indices.tolist()


def _prompt_metrics(
    candidate: torch.Tensor,
    control: torch.Tensor,
    reference: torch.Tensor,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        start = record["captured_start"]
        end = record["captured_end"]
        result.append(
            {
                "prompt_index": record["prompt_index"],
                "eligible_user_content_tokens": end - start,
                "candidate_mixed_gptq": expert_gate._comparison(
                    candidate[start:end], reference[start:end]
                ),
                "equal_storage_rtn": expert_gate._comparison(
                    control[start:end], reference[start:end]
                ),
            }
        )
    return result


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def _write_result(path: Path, result: dict[str, Any]) -> None:
    text = json.dumps(result, indent=2)
    print(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=expert_gate.DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bits", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--mse-grid", type=int, default=100)
    parser.add_argument("--no-mse", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=160)
    parser.add_argument("--max-calibration-tokens", type=int, default=256)
    parser.add_argument("--min-routed-calibration-hits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if (
        args.group_size <= 0
        or args.block_size <= 0
        or args.mse_grid <= 0
        or args.max_calibration_tokens < 0
        or args.min_routed_calibration_hits < 1
    ):
        parser.error("sizes and MSE grid must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this bounded gate requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    script_path = Path(__file__).resolve()
    imported_gate_path = Path(expert_gate.__file__).resolve()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    reader = expert_gate.CheckpointReader(source)
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
    started = time.perf_counter()
    attention, rotary, input_norm, post_norm, router = (
        expert_gate._load_layer0_front(reader, config, device, dtype)
    )
    embedding_cache: dict[int, torch.Tensor] = {}
    with reader.embedding_handle() as embedding_handle:
        calibration, calibration_stats = _capture_split(
            expert_gate.CALIBRATION_PROMPTS,
            tokenizer=tokenizer,
            embedding_handle=embedding_handle,
            embedding_cache=embedding_cache,
            attention=attention,
            rotary=rotary,
            input_norm_weight=input_norm,
            post_attention_norm_weight=post_norm,
            router_weight=router,
            config=config,
            device=device,
            dtype=dtype,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        holdout, holdout_stats = _capture_split(
            expert_gate.HOLDOUT_PROMPTS,
            tokenizer=tokenizer,
            embedding_handle=embedding_handle,
            embedding_cache=embedding_cache,
            attention=attention,
            rotary=rotary,
            input_norm_weight=input_norm,
            post_attention_norm_weight=post_norm,
            router_weight=router,
            config=config,
            device=device,
            dtype=dtype,
            max_prompt_tokens=args.max_prompt_tokens,
        )
    capture_seconds = time.perf_counter() - started

    del attention, rotary, input_norm, post_norm, router, embedding_cache
    gc.collect()
    torch.cuda.empty_cache()

    holdout_input = holdout["expert_input"]
    holdout_indices = holdout["route_indices"]
    holdout_weights = holdout["route_weights"]
    holdout_tokens = holdout_input.shape[0]
    reference_moe = torch.zeros((holdout_tokens, config.hidden_size), dtype=torch.float32)
    candidate_moe = torch.zeros_like(reference_moe)
    rtn_moe = torch.zeros_like(reference_moe)

    routed_experts = sorted(torch.unique(holdout_indices).tolist())
    expert_results: dict[str, Any] = {}
    quantization_started = time.perf_counter()

    for ordinal, expert in enumerate(routed_experts, start=1):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        gate = reader.tensor(
            f"{prefix}.gate_proj.weight", device=device, dtype=torch.float32
        )
        up = reader.tensor(
            f"{prefix}.up_proj.weight", device=device, dtype=torch.float32
        )
        down = reader.tensor(
            f"{prefix}.down_proj.weight", device=device, dtype=torch.float32
        )

        expert_started = time.perf_counter()
        rtn_gate = expert_gate._rtn_quantize(
            gate,
            nbits=args.bits,
            group_size=args.group_size,
            mse=not args.no_mse,
            mse_grid=args.mse_grid,
        )

        calibration_route_mask = torch.any(
            calibration["route_indices"] == expert, dim=-1
        )
        routed_calibration_rows = calibration["expert_input"][
            calibration_route_mask
        ]
        calibration_sample_cpu, calibration_indices = _sample_calibration_rows(
            routed_calibration_rows,
            args.max_calibration_tokens,
            args.seed + int(expert),
        )
        calibration_index_sha = hashlib.sha256(
            json.dumps(calibration_indices, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        use_gptq = (
            calibration_sample_cpu.shape[0] >= args.min_routed_calibration_hits
        )
        input_damp: float | None = None
        input_dead_fraction: float | None = None
        down_damp: float | None = None
        down_dead_fraction: float | None = None
        calibration_input = None
        calibration_hidden = None
        input_inverse = None
        down_inverse = None
        gptq_gate = None
        gptq_up = None
        gptq_down = None
        if use_gptq:
            calibration_input = calibration_sample_cpu.to(device=device)
            input_inverse, input_damp, input_dead_fraction = (
                expert_gate._input_hessian(calibration_input, args.percdamp)
            )
            calibration_hidden = F.silu(F.linear(calibration_input, gate)) * F.linear(
                calibration_input, up
            )
            down_inverse, down_damp, down_dead_fraction = expert_gate._input_hessian(
                calibration_hidden, args.percdamp
            )
            gptq_gate = expert_gate._gptq_quantize(
                gate,
                input_inverse,
                nbits=args.bits,
                group_size=args.group_size,
                block_size=args.block_size,
                mse=not args.no_mse,
                mse_grid=args.mse_grid,
            )
            gptq_up = expert_gate._gptq_quantize(
                up,
                input_inverse,
                nbits=args.bits,
                group_size=args.group_size,
                block_size=args.block_size,
                mse=not args.no_mse,
                mse_grid=args.mse_grid,
            )
            gptq_down = expert_gate._gptq_quantize(
                down,
                down_inverse,
                nbits=args.bits,
                group_size=args.group_size,
                block_size=args.block_size,
                mse=not args.no_mse,
                mse_grid=args.mse_grid,
            )
        rtn_up = expert_gate._rtn_quantize(
            up,
            nbits=args.bits,
            group_size=args.group_size,
            mse=not args.no_mse,
            mse_grid=args.mse_grid,
        )
        rtn_down = expert_gate._rtn_quantize(
            down,
            nbits=args.bits,
            group_size=args.group_size,
            mse=not args.no_mse,
            mse_grid=args.mse_grid,
        )

        positions, slots = torch.where(holdout_indices == expert)
        selected_input = holdout_input[positions].to(device=device)
        selected_route_weight = holdout_weights[positions, slots, None]
        _, _, reference_output = expert_gate._expert_output(
            selected_input, gate, up, down
        )
        _, _, rtn_output = expert_gate._expert_output(
            selected_input, rtn_gate, rtn_up, rtn_down
        )
        if use_gptq:
            assert gptq_gate is not None and gptq_up is not None and gptq_down is not None
            _, _, candidate_output = expert_gate._expert_output(
                selected_input, gptq_gate, gptq_up, gptq_down
            )
            candidate_method = "routed-activation-gptq"
        else:
            candidate_output = rtn_output
            candidate_method = "equal-storage-rtn-fallback-insufficient-routes"
        torch.cuda.synchronize(device)
        expert_seconds = time.perf_counter() - expert_started

        # CPU accumulation is deterministic and avoids CUDA index-add atomics.
        route_weight_cpu = selected_route_weight.float()
        reference_moe[positions] += reference_output.cpu() * route_weight_cpu
        candidate_moe[positions] += candidate_output.cpu() * route_weight_cpu
        rtn_moe[positions] += rtn_output.cpu() * route_weight_cpu

        shapes = (tuple(gate.shape), tuple(up.shape), tuple(down.shape))
        packed_bytes = sum(
            expert_gate._runtime_bytes(shape, args.bits, args.group_size)
            for shape in shapes
        )
        parameters = sum(math.prod(shape) for shape in shapes)
        candidate_selected_metrics = expert_gate._comparison(
            candidate_output, reference_output
        )
        rtn_selected_metrics = expert_gate._comparison(rtn_output, reference_output)
        expert_results[str(expert)] = {
            "holdout_route_hits": int(positions.numel()),
            "candidate_method": candidate_method,
            "routed_calibration_hits_available": int(routed_calibration_rows.shape[0]),
            "calibration_rows_used": int(calibration_sample_cpu.shape[0]),
            "calibration_local_indices_sha256": calibration_index_sha,
            "input_hessian_damp": input_damp,
            "input_hessian_dead_diagonal_fraction": input_dead_fraction,
            "down_hessian_damp": down_damp,
            "down_hessian_dead_diagonal_fraction": down_dead_fraction,
            "runtime_packed_bytes": packed_bytes,
            "effective_bits_per_weight": packed_bytes * 8.0 / parameters,
            "quantization_and_evaluation_seconds": expert_seconds,
            "candidate_selected_output": candidate_selected_metrics,
            "equal_storage_rtn_selected_output": rtn_selected_metrics,
            "relative_selected_output_nrmse_improvement_vs_rtn": (
                rtn_selected_metrics["normalized_rmse"]
                - candidate_selected_metrics["normalized_rmse"]
            )
            / max(rtn_selected_metrics["normalized_rmse"], 1e-12),
        }
        print(
            f"[{ordinal:03d}/{len(routed_experts):03d}] expert {expert:03d}: "
            f"hits={positions.numel():4d}, "
            f"method={candidate_method}, "
            f"candidate_nrmse={candidate_selected_metrics['normalized_rmse']:.6f}, "
            f"rtn_nrmse={rtn_selected_metrics['normalized_rmse']:.6f}, "
            f"seconds={expert_seconds:.2f}",
            flush=True,
        )

        del (
            gate,
            up,
            down,
            calibration_input,
            calibration_hidden,
            input_inverse,
            down_inverse,
            gptq_gate,
            gptq_up,
            gptq_down,
            rtn_gate,
            rtn_up,
            rtn_down,
            selected_input,
            reference_output,
            candidate_output,
            rtn_output,
        )
        gc.collect()
        torch.cuda.empty_cache()

    quantization_seconds = time.perf_counter() - quantization_started
    reference_layer = holdout["residual"] + reference_moe
    candidate_layer = holdout["residual"] + candidate_moe
    rtn_layer = holdout["residual"] + rtn_moe

    routed_candidate = expert_gate._comparison(candidate_moe, reference_moe)
    routed_rtn = expert_gate._comparison(rtn_moe, reference_moe)
    layer_candidate = expert_gate._comparison(candidate_layer, reference_layer)
    layer_rtn = expert_gate._comparison(rtn_layer, reference_layer)
    prompt_moe = _prompt_metrics(
        candidate_moe,
        rtn_moe,
        reference_moe,
        holdout_stats["prompt_records"],
    )
    prompt_layer = _prompt_metrics(
        candidate_layer,
        rtn_layer,
        reference_layer,
        holdout_stats["prompt_records"],
    )

    relative_improvement = (
        routed_rtn["normalized_rmse"] - routed_candidate["normalized_rmse"]
    ) / max(routed_rtn["normalized_rmse"], 1e-12)
    thresholds = {
        "routed_moe_output_nrmse_max": 0.30,
        "routed_moe_output_cosine_min": 0.95,
        "top8_identity_rate_min": 1.0,
        "minimum_relative_nrmse_improvement_vs_equal_storage_rtn": 0.20,
    }
    route_metrics = {
        "top8_identity_rate": 1.0,
        "top8_set_jaccard": 1.0,
        "routing_weight_rmse": 0.0,
        "routing_weight_max_abs": 0.0,
        "explanation": (
            "The source BF16 router is retained unchanged in BF16 reference, "
            "mixed GPTQ candidate, and equal-storage RTN control; the captured route "
            "identities and weights are shared exactly by construction."
        ),
    }
    absolute_pass = (
        routed_candidate["normalized_rmse"]
        <= thresholds["routed_moe_output_nrmse_max"]
        and routed_candidate["cosine_similarity"]
        >= thresholds["routed_moe_output_cosine_min"]
        and route_metrics["top8_identity_rate"]
        >= thresholds["top8_identity_rate_min"]
    )
    comparative_pass = (
        relative_improvement
        >= thresholds["minimum_relative_nrmse_improvement_vs_equal_storage_rtn"]
    )

    example_prefix = "model.layers.0.mlp.experts.0"
    example_shapes = (
        tuple(
            reader.tensor(
                f"{example_prefix}.gate_proj.weight",
                device=torch.device("cpu"),
                dtype=torch.float32,
            ).shape
        ),
        tuple(
            reader.tensor(
                f"{example_prefix}.up_proj.weight",
                device=torch.device("cpu"),
                dtype=torch.float32,
            ).shape
        ),
        tuple(
            reader.tensor(
                f"{example_prefix}.down_proj.weight",
                device=torch.device("cpu"),
                dtype=torch.float32,
            ).shape
        ),
    )
    per_expert_runtime_bytes = sum(
        expert_gate._runtime_bytes(shape, args.bits, args.group_size)
        for shape in example_shapes
    )
    per_expert_parameters = sum(math.prod(shape) for shape in example_shapes)
    calibration_selections_sha = hashlib.sha256(
        json.dumps(
            {
                expert: {
                    "rows": record["calibration_rows_used"],
                    "indices_sha256": record["calibration_local_indices_sha256"],
                }
                for expert, record in expert_results.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    result: dict[str, Any] = {
        "experiment": "gemq_style_router_weighted_complete_layer0_gate",
        "scope": (
            "complete layer-0 routed MoE output for every holdout user-content "
            "token; this is not a full-model generation certificate"
        ),
        "source_model": str(source),
        "source_revision": provenance.get("resolved_revision"),
        "source_architecture": getattr(config, "architectures", None),
        "gemq_source_commit": expert_gate.GEMQ_COMMIT,
        "reproducibility": {
            "seed": args.seed,
            "deterministic_algorithms": True,
            "calibration_prompt_sha256": expert_gate._prompt_fingerprint(
                expert_gate.CALIBRATION_PROMPTS
            ),
            "holdout_prompt_sha256": expert_gate._prompt_fingerprint(
                expert_gate.HOLDOUT_PROMPTS
            ),
            "all_expert_calibration_selections_sha256": calibration_selections_sha,
            "script_sha256": _sha256_file(script_path),
            "imported_expert_gate_sha256": _sha256_file(imported_gate_path),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "platform": sys.platform,
        },
        "layer": 0,
        "candidate": (
            f"source BF16 router plus routed-calibrated {args.bits}-bit GPTQ "
            "experts with equal-storage RTN fallback below the minimum route count"
        ),
        "equal_storage_control": (
            f"same source BF16 router plus uniform {args.bits}-bit optimized RTN experts"
        ),
        "reference": (
            "source BF16-valued router and expert weights evaluated with FP32 "
            "expert matmul accumulation to isolate weight reconstruction error"
        ),
        "quantization": {
            "bits": args.bits,
            "group_size": args.group_size,
            "block_size": args.block_size,
            "percdamp": args.percdamp,
            "mse_range_search": not args.no_mse,
            "mse_grid": args.mse_grid,
            "packing_model": "GemLite int32 columns + FP16 scale and zero",
            "router_storage": "unchanged source BF16",
            "per_expert_parameters": per_expert_parameters,
            "per_expert_runtime_packed_bytes": per_expert_runtime_bytes,
            "all_experts_runtime_packed_bytes": (
                per_expert_runtime_bytes * config.num_experts
            ),
            "effective_expert_bits_per_weight": (
                per_expert_runtime_bytes * 8.0 / per_expert_parameters
            ),
        },
        "calibration_capture": calibration_stats,
        "holdout_capture": holdout_stats,
        "minimum_routed_calibration_hits_for_gptq": args.min_routed_calibration_hits,
        "calibration_sampling": (
            "per-expert deterministic seeded samples of only the real layer-0 "
            "inputs routed to that expert; insufficiently observed experts fall "
            "back to the equal-storage RTN representation without holdout leakage"
        ),
        "routed_experts_evaluated": routed_experts,
        "routed_experts_evaluated_count": len(routed_experts),
        "route_metrics": route_metrics,
        "complete_routed_moe_output": {
            "candidate_mixed_gptq": routed_candidate,
            "equal_storage_rtn": routed_rtn,
            "relative_gptq_nrmse_improvement_vs_rtn": relative_improvement,
        },
        "decoder_layer_output_after_residual": {
            "candidate_mixed_gptq": layer_candidate,
            "equal_storage_rtn": layer_rtn,
        },
        "per_prompt_routed_moe_output": prompt_moe,
        "per_prompt_layer_output_after_residual": prompt_layer,
        "per_prompt_summary": {
            "candidate_routed_moe_nrmse": _summary(
                [
                    record["candidate_mixed_gptq"]["normalized_rmse"]
                    for record in prompt_moe
                ]
            ),
            "rtn_routed_moe_nrmse": _summary(
                [
                    record["equal_storage_rtn"]["normalized_rmse"]
                    for record in prompt_moe
                ]
            ),
            "candidate_layer_output_nrmse": _summary(
                [
                    record["candidate_mixed_gptq"]["normalized_rmse"]
                    for record in prompt_layer
                ]
            ),
            "rtn_layer_output_nrmse": _summary(
                [
                    record["equal_storage_rtn"]["normalized_rmse"]
                    for record in prompt_layer
                ]
            ),
        },
        "expert_results": expert_results,
        "gate_thresholds": thresholds,
        "gate_components": {
            "absolute_quality": absolute_pass,
            "comparative_advantage": comparative_pass,
        },
        "capture_seconds": capture_seconds,
        "quantization_and_evaluation_seconds": quantization_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_host_working_set_bytes": expert_gate._peak_host_working_set_bytes(),
        "status": (
            "pass-router-weighted-layer0-gate"
            if absolute_pass and comparative_pass
            else "reject-router-weighted-layer0-gate"
        ),
        "next_gate": (
            "eligible for a bounded multi-layer/full-pack experiment"
            if absolute_pass and comparative_pass
            else "do not full-pack this codec configuration"
        ),
    }
    _write_result(output, result)
    if not (absolute_pass and comparative_pass):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
