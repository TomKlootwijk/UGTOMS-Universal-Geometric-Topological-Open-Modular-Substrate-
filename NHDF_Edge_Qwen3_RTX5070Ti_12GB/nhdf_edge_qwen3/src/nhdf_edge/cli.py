"""Command-line interface for conversion, verification and feasibility checks."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from .checkpoint import pack_checkpoint
from .config import load_config
from .format import PackReader
from .metrics import context_sweep, estimate, residual_fraction_sweep
from .quantize import QuantizationPolicy, dequantize_tensor, quantize_tensor, verify_parity
from .runtime.doctor import run_doctor


def _json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def command_estimate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    result = estimate(cfg, context_tokens=args.context)
    payload = {
        "estimate": result.to_dict(),
        "residual_fraction_sweep": residual_fraction_sweep(cfg, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        "context_sweep": context_sweep(cfg, [4096, 8192, 16384, 32768]),
        "status": "analytical-not-measured",
    }
    _json(payload)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def command_pack(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    manifest = pack_checkpoint(
        args.source,
        args.output,
        cfg,
        include=args.include,
        exclude=args.exclude,
        max_tensors=args.max_tensors,
        hessian_path=args.hessian,
    )
    print(manifest)
    return 0


def command_verify(args: argparse.Namespace) -> int:
    reader = PackReader(args.pack, verify_crc=True)
    result = reader.verify_all()
    parity_failures = []
    names = reader.names() if args.parity_all else reader.names()[: args.parity_sample]
    for name in names:
        check = verify_parity(reader.load(name))
        if not check["ok"]:
            parity_failures.append({"tensor": name, **check})
    result["parity_checked"] = len(names)
    result["parity_failures"] = parity_failures
    result["ok"] = bool(result["ok"] and not parity_failures)
    _json(result)
    return 0 if result["ok"] else 2


def command_doctor(args: argparse.Namespace) -> int:
    report = run_doctor(load_config(args.config))
    _json(report.to_dict())
    return 0 if report.verdict == "ready-for-benchmark" else 1


def command_smoke(args: argparse.Namespace) -> int:
    generator = torch.Generator().manual_seed(args.seed)
    weight = torch.randn(args.rows, args.cols, generator=generator)
    base_policy = QuantizationPolicy(
        base_bits=2,
        group_size=256,
        residual_fraction=0.0,
        iterations=3,
    )
    branch_policy = QuantizationPolicy(
        base_bits=2,
        group_size=256,
        residual_fraction=args.residual_fraction,
        iterations=3,
    )
    base = quantize_tensor(weight, base_policy, name="smoke.base")
    branch = quantize_tensor(weight, branch_policy, name="smoke.branch")
    base_mse = torch.mean((weight - dequantize_tensor(base)) ** 2).item()
    branch_mse = torch.mean((weight - dequantize_tensor(branch)) ** 2).item()
    payload = {
        "shape": list(weight.shape),
        "base_mse": base_mse,
        "branch_mse": branch_mse,
        "mse_reduction_percent": 100.0 * (1.0 - branch_mse / base_mse),
        "branch_effective_bpp": branch.stats["effective_bits_per_weight"],
        "zero_set_max_abs": branch.stats["weighted_zero_set_max_abs"],
        "parity": verify_parity(branch),
    }
    _json(payload)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nhdf-edge")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("estimate", help="print analytical size/throughput projections")
    p.add_argument("--config")
    p.add_argument("--context", type=int)
    p.add_argument("--output")
    p.set_defaults(func=command_estimate)

    p = sub.add_parser("pack", help="stream-convert a local Hugging Face checkpoint")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--config")
    p.add_argument("--include")
    p.add_argument("--exclude")
    p.add_argument("--max-tensors", type=int)
    p.add_argument("--hessian", help="optional safetensors file of input second moments")
    p.set_defaults(func=command_pack)

    p = sub.add_parser("verify", help="verify CRC32 and one-bit payload parity")
    p.add_argument("pack")
    p.add_argument("--parity-sample", type=int, default=16)
    p.add_argument("--parity-all", action="store_true")
    p.set_defaults(func=command_verify)

    p = sub.add_parser("doctor", help="inspect CUDA/VRAM readiness")
    p.add_argument("--config")
    p.set_defaults(func=command_doctor)

    p = sub.add_parser("smoke", help="run a deterministic synthetic quantization test")
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--cols", type=int, default=1024)
    p.add_argument("--residual-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output")
    p.set_defaults(func=command_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
