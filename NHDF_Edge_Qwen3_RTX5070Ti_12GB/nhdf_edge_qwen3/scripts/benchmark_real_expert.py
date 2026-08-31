#!/usr/bin/env python3
"""Compare one real BF16 Qwen expert with its packed reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F

from nhdf_edge.format import PackReader
from nhdf_edge.quantize import (
    QuantizationPolicy,
    dequantize_rows,
    dequantize_tensor,
    quantize_tensor,
)


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.float()
    reference = reference.float()
    diff = actual - reference
    rmse = diff.square().mean().sqrt()
    reference_rms = reference.square().mean().sqrt()
    return {
        "rmse": float(rmse),
        "reference_rms": float(reference_rms),
        "normalized_rmse": float(rmse / reference_rms.clamp_min(1e-12)),
        "max_abs": float(diff.abs().max()),
        "cosine_similarity": float(
            F.cosine_similarity(actual.reshape(1, -1), reference.reshape(1, -1))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("pack_dir")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--local-ablations", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    source = Path(args.source_dir)
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]

    def source_tensor(name: str) -> torch.Tensor:
        with safe_open(str(source / weight_map[name]), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).float().clone()

    source_prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}"
    gate = source_tensor(f"{source_prefix}.gate_proj.weight")
    up = source_tensor(f"{source_prefix}.up_proj.weight")
    down = source_tensor(f"{source_prefix}.down_proj.weight")
    intermediate, hidden = gate.shape

    reader = PackReader(args.pack_dir)
    packed_prefix = f"model.layers.{args.layer}.mlp.experts"
    gate_up_pack = reader.load(f"{packed_prefix}.gate_up_proj")
    down_pack = reader.load(f"{packed_prefix}.down_proj")
    gate_start = args.expert * 2 * intermediate
    down_start = args.expert * hidden
    reconstructed_gate_up = dequantize_rows(
        gate_up_pack,
        torch.arange(gate_start, gate_start + 2 * intermediate),
        dtype=torch.float32,
    )
    reconstructed_down = dequantize_rows(
        down_pack,
        torch.arange(down_start, down_start + hidden),
        dtype=torch.float32,
    )

    generator = torch.Generator().manual_seed(args.seed)
    inputs = torch.randn(args.tokens, hidden, generator=generator)
    reference_gate_up = F.linear(inputs, torch.cat((gate, up)))
    packed_gate_up = F.linear(inputs, reconstructed_gate_up)
    reference_gate, reference_up = reference_gate_up.chunk(2, dim=-1)
    packed_gate, packed_up = packed_gate_up.chunk(2, dim=-1)
    reference_hidden = F.silu(reference_gate) * reference_up
    packed_hidden = F.silu(packed_gate) * packed_up
    reference_output = F.linear(reference_hidden, down)
    packed_output = F.linear(packed_hidden, reconstructed_down)

    def evaluate_weights(gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> dict[str, object]:
        candidate_gate_up = F.linear(inputs, gate_up_weight)
        candidate_gate, candidate_up = candidate_gate_up.chunk(2, dim=-1)
        candidate_hidden = F.silu(candidate_gate) * candidate_up
        candidate_output = F.linear(candidate_hidden, down_weight)
        return {
            "gate_up_weight": _metrics(gate_up_weight, torch.cat((gate, up))),
            "down_weight": _metrics(down_weight, down),
            "expert_output": _metrics(candidate_output, reference_output),
        }

    local_ablations: dict[str, object] | None = None
    if args.local_ablations:
        local_ablations = {}
        for label, policy in {
            "plain_2bit": QuantizationPolicy(base_bits=2, group_size=256, residual_fraction=0.0),
            "nhdf_2bit_residual_0_15": QuantizationPolicy(
                base_bits=2,
                group_size=256,
                residual_fraction=0.15,
            ),
            "plain_4bit": QuantizationPolicy(base_bits=4, group_size=256, residual_fraction=0.0),
        }.items():
            quantized_gate_up = quantize_tensor(torch.cat((gate, up)), policy, name=f"{label}.gate_up")
            quantized_down = quantize_tensor(down, policy, name=f"{label}.down")
            comparison = evaluate_weights(
                dequantize_tensor(quantized_gate_up),
                dequantize_tensor(quantized_down),
            )
            comparison["packed_payload_bytes"] = int(
                quantized_gate_up.stats["packed_bytes"] + quantized_down.stats["packed_bytes"]
            )
            local_ablations[label] = comparison

    provenance_path = source / "NHDF_SOURCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
    result = {
        "source_revision": (provenance or {}).get("resolved_revision"),
        "layer": args.layer,
        "expert": args.expert,
        "tokens": args.tokens,
        "seed": args.seed,
        "shape": {
            "hidden": hidden,
            "intermediate": intermediate,
        },
        "gate_up_weight": _metrics(reconstructed_gate_up, torch.cat((gate, up))),
        "down_weight": _metrics(reconstructed_down, down),
        "gate_up_projection": _metrics(packed_gate_up, reference_gate_up),
        "activated_hidden": _metrics(packed_hidden, reference_hidden),
        "expert_output": _metrics(packed_output, reference_output),
        "isolated_single_expert_ablations": local_ablations,
        "status": "measured-real-bf16-versus-packed-expert",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
