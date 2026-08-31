#!/usr/bin/env python3
"""Benchmark and validate the fused packed GEMV on the target GPU."""
from __future__ import annotations

import argparse
import json
import statistics

import torch
from torch.nn import functional as F

from nhdf_edge.quantize import QuantizationPolicy, dequantize_tensor, quantize_tensor
from nhdf_edge.runtime.cuda_backend import require
from nhdf_edge.runtime.modules import NHDFPackedLinear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--bits", type=int, choices=[2, 4], default=2)
    parser.add_argument("--residual-fraction", type=float, default=0.15)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
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
    module = NHDFPackedLinear(packed).cuda().eval()
    x = torch.randn(args.batch, args.cols, generator=generator, dtype=torch.float16, device="cuda")

    reference_weight = dequantize_tensor(packed, dtype=torch.float16).cuda()
    expected = F.linear(x, reference_weight)
    actual = module(x)
    max_abs = (actual - expected).abs().max().item()
    max_rel = ((actual - expected).abs() / expected.abs().clamp_min(1e-4)).max().item()
    del reference_weight, expected
    torch.cuda.empty_cache()

    for _ in range(args.warmup):
        module(x)
    torch.cuda.synchronize()
    timings = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        module(x)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))

    median_ms = statistics.median(timings)
    packed_bytes = sum(t.numel() * t.element_size() for t in packed.tensor_dict().values())
    effective_gbps = packed_bytes * args.batch / (median_ms / 1000.0) / 1e9
    result = {
        "device": torch.cuda.get_device_name(0),
        "shape": [args.rows, args.cols],
        "bits": args.bits,
        "residual_fraction": residual,
        "batch": args.batch,
        "median_ms": median_ms,
        "p10_ms": sorted(timings)[max(0, int(0.10 * len(timings)) - 1)],
        "p90_ms": sorted(timings)[min(len(timings) - 1, int(0.90 * len(timings)))],
        "effective_weight_read_gbps": effective_gbps,
        "max_abs_error_vs_dequantized_fp16": max_abs,
        "max_relative_error_vs_dequantized_fp16": max_rel,
        "status": "measured-kernel-microbenchmark",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
