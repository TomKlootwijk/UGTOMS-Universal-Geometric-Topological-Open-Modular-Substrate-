#!/usr/bin/env python3
"""Validate the configured HQQ int8 KV cache on the target GPU."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import QuantizedCache

from nhdf_edge.config import load_config


def _tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--config", default="configs/qwen3_30b_a3b_edge12.yaml")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.tokens < 1:
        raise SystemExit("--tokens must be positive")

    hf_config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    profile = load_config(args.config)
    target = profile.target
    if target.kv_bits != 8:
        raise SystemExit("this benchmark currently targets the configured 8-bit HQQ cache")
    cache = QuantizedCache(
        backend="hqq",
        config=hf_config,
        nbits=target.kv_bits,
        axis_key=0,
        axis_value=0,
        q_group_size=target.kv_group_size,
        residual_length=target.kv_residual_length,
    )
    generator = torch.Generator(device="cuda").manual_seed(20260831)
    reference_keys: list[torch.Tensor] = []
    reference_values: list[torch.Tensor] = []
    timings_ms: list[float] = []
    returned_keys = returned_values = None
    for _ in range(args.tokens):
        shape = (1, hf_config.num_key_value_heads, 1, hf_config.head_dim)
        key = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16)
        value = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16)
        reference_keys.append(key)
        reference_values.append(value)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        returned_keys, returned_values = cache.update(key, value, 0)
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))

    expected_keys = torch.cat(reference_keys, dim=-2)
    expected_values = torch.cat(reference_values, dim=-2)
    error = torch.cat(
        [(returned_keys - expected_keys).float().reshape(-1), (returned_values - expected_values).float().reshape(-1)]
    )
    reference = torch.cat([expected_keys.float().reshape(-1), expected_values.float().reshape(-1)])
    layer = cache.layers[0]
    one_layer_bytes = (
        _tensor_bytes(layer._quantized_keys)
        + _tensor_bytes(layer._quantized_values)
        + _tensor_bytes(layer.keys)
        + _tensor_bytes(layer.values)
    )
    result = {
        "device": torch.cuda.get_device_name(0),
        "tokens": args.tokens,
        "layers": hf_config.num_hidden_layers,
        "kv_heads": hf_config.num_key_value_heads,
        "head_dim": hf_config.head_dim,
        "bits": target.kv_bits,
        "group_size": target.kv_group_size,
        "residual_length": target.kv_residual_length,
        "one_layer_stored_bytes": one_layer_bytes,
        "extrapolated_all_layer_bytes": one_layer_bytes * hf_config.num_hidden_layers,
        "median_update_ms": statistics.median(timings_ms),
        "first_update_ms": timings_ms[0],
        "p90_update_ms": sorted(timings_ms)[min(len(timings_ms) - 1, int(0.9 * len(timings_ms)))],
        "p99_update_ms": sorted(timings_ms)[min(len(timings_ms) - 1, int(0.99 * len(timings_ms)))],
        "max_update_ms": max(timings_ms),
        "max_update_token_index": timings_ms.index(max(timings_ms)),
        "max_abs_error": error.abs().max().item(),
        "rmse": error.square().mean().sqrt().item(),
        "normalized_rmse": error.square().mean().sqrt().item()
        / max(reference.square().mean().sqrt().item(), 1e-12),
        "status": "measured-hqq-int8-cache",
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
