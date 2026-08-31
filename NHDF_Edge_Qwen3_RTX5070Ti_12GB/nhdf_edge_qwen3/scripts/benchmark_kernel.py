#!/usr/bin/env python3
"""Benchmark and validate the fused packed GEMV on the target GPU."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from nhdf_edge.quantize import QuantizationPolicy, dequantize_tensor, quantize_tensor, residual_mask
from nhdf_edge.runtime.cuda_backend import require
from nhdf_edge.runtime.modules import PackedMatrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--bits", type=int, choices=[2, 4], default=2)
    parser.add_argument("--residual-fraction", type=float, default=0.15)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--row-count", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-abs-tolerance", type=float, default=0.5)
    parser.add_argument("--normalized-rmse-tolerance", type=float, default=1e-3)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.99999)
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    require()
    residual = args.residual_fraction if args.bits == 2 else 0.0
    generator = torch.Generator().manual_seed(123)
    weight = torch.randn(args.rows, args.cols, generator=generator)
    packed = quantize_tensor(
        weight,
        QuantizationPolicy(base_bits=args.bits, group_size=256, residual_fraction=residual),
        name="benchmark.weight",
    )
    row_count = args.rows - args.row_offset if args.row_count is None else args.row_count
    if args.row_offset < 0 or row_count <= 0 or args.row_offset + row_count > args.rows:
        raise SystemExit("row interval must be non-empty and inside the matrix")
    module = PackedMatrix(packed).cuda().eval()
    # A CPU generator cannot directly drive a CUDA allocation on current
    # PyTorch.  Generate deterministically on CPU, then transfer to the target.
    x = torch.randn(args.batch, args.cols, generator=generator, dtype=torch.float16).cuda()

    reference_weight = dequantize_tensor(packed, dtype=torch.float16)[
        args.row_offset : args.row_offset + row_count
    ].cuda()
    expected = F.linear(x, reference_weight)
    actual = module.project(x, row_offset=args.row_offset, row_count=row_count)
    error = (actual - expected).float()
    expected_float = expected.float()
    max_abs = error.abs().max().item()
    max_rel = (error.abs() / expected_float.abs().clamp_min(1e-4)).max().item()
    rmse = error.square().mean().sqrt().item()
    reference_rms = expected_float.square().mean().sqrt().item()
    normalized_rmse = rmse / max(reference_rms, 1e-12)
    cosine = F.cosine_similarity(
        actual.float().reshape(1, -1), expected_float.reshape(1, -1), dim=-1
    ).item()
    del reference_weight, expected
    torch.cuda.empty_cache()

    for _ in range(args.warmup):
        module.project(x, row_offset=args.row_offset, row_count=row_count)
    torch.cuda.synchronize()
    timings = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module.project(x, row_offset=args.row_offset, row_count=row_count)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))

    median_ms = statistics.median(timings)
    serialized_bytes = sum(t.numel() * t.element_size() for t in packed.tensor_dict().values())
    groups = row_count * packed.groups_per_row
    runtime_bytes = row_count * packed.padded_cols * args.bits // 8
    runtime_bytes += groups * 4  # FP16 mean + FP16 scale.
    if packed.residual_mask_words.numel():
        first_group = args.row_offset * packed.groups_per_row
        selected = int(residual_mask(packed)[first_group : first_group + groups].sum().item())
        mask_words = (groups + 31) // 32
        runtime_bytes += mask_words * 8  # int32 mask + int32 rank prefix.
        runtime_bytes += selected * (packed.policy.group_size // 8 + 2)
    effective_gbps = runtime_bytes * args.batch / (median_ms / 1000.0) / 1e9
    equivalence_ok = bool(
        max_abs <= args.max_abs_tolerance
        and normalized_rmse <= args.normalized_rmse_tolerance
        and cosine >= args.min_cosine_similarity
    )
    result = {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "shape": [args.rows, args.cols],
        "row_offset": args.row_offset,
        "row_count": row_count,
        "bits": args.bits,
        "residual_fraction": residual,
        "batch": args.batch,
        "median_ms": median_ms,
        "p10_ms": sorted(timings)[max(0, int(0.10 * len(timings)) - 1)],
        "p90_ms": sorted(timings)[min(len(timings) - 1, int(0.90 * len(timings)))],
        "serialized_tensor_bytes": serialized_bytes,
        "estimated_kernel_bytes_per_batch": runtime_bytes,
        "effective_weight_read_gbps": effective_gbps,
        "max_abs_error_vs_dequantized_fp16": max_abs,
        "max_relative_error_vs_dequantized_fp16": max_rel,
        "rmse_vs_dequantized_fp16": rmse,
        "normalized_rmse_vs_dequantized_fp16": normalized_rmse,
        "cosine_similarity_vs_dequantized_fp16": cosine,
        "equivalence_thresholds": {
            "max_abs": args.max_abs_tolerance,
            "normalized_rmse": args.normalized_rmse_tolerance,
            "min_cosine_similarity": args.min_cosine_similarity,
        },
        "equivalence_ok": equivalence_ok,
        "status": "pass" if equivalence_ok else "fail",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if not equivalence_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
