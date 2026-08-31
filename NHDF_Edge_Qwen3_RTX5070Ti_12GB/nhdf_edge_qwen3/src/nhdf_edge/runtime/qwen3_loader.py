"""Experimental full-model integration for Qwen3-MoE.

This loader constructs the Hugging Face model on the meta device, replaces
matrix-bearing modules with NHDF packed modules, installs raw router/norm
parameters, and checks that no meta parameters remain.  It requires the optional
CUDA extension for practical execution.  The implementation is intentionally
kept separate from the pack format so Transformers version drift can be adapted
without changing serialized weights.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..format import PackReader
from .cuda_backend import available as cuda_backend_available
from .modules import NHDFPackedEmbedding, NHDFPackedLinear, NHDFQwen3Experts


def _optional_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Full-model loading requires `pip install -e '.[runtime]'` "
            "(Transformers and Accelerate)."
        ) from exc
    return init_empty_weights, AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _get_child(obj: Any, part: str) -> Any:
    if part.isdigit():
        return obj[int(part)]
    return getattr(obj, part)


def _get_path(root: Any, path: str) -> Any:
    current = root
    if not path:
        return current
    for part in path.split("."):
        current = _get_child(current, part)
    return current


def _set_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    parent = _get_path(root, ".".join(parts[:-1]))
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _raw_parameter(reader: PackReader, name: str, device: torch.device, dtype: torch.dtype) -> nn.Parameter:
    packed = reader.load(name, device="cpu")
    if packed.policy.mode != "raw":
        raise ValueError(f"expected raw tensor for {name}")
    tensor = packed.raw
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    return nn.Parameter(tensor.to(device=device), requires_grad=False)


def _load_packed(reader: PackReader, name: str, device: torch.device):
    return reader.load(name, device=device)


def load_tokenizer(pack_dir: str | Path):
    """Load the tokenizer copied into ``hf_metadata`` during conversion."""

    _, _, _, AutoTokenizer = _optional_imports()
    metadata = Path(pack_dir) / "hf_metadata"
    return AutoTokenizer.from_pretrained(metadata, local_files_only=True)


def load_qwen3_moe(
    pack_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    verify_crc: bool = True,
    require_cuda_extension: bool = True,
    allow_unvalidated: bool = False,
):
    """Build a Qwen3-MoE causal LM directly from an NHDF pack.

    Parameters
    ----------
    pack_dir:
        Converted pack containing ``manifest.json`` and ``hf_metadata``.
    device:
        Runtime device.  The edge profile is designed for one CUDA GPU.
    dtype:
        Activation/raw-parameter dtype.  The reference CUDA kernel supports
        float16 and float32 inputs; float16 is the intended profile.
    verify_crc:
        Verify every per-tensor file as it is loaded.
    require_cuda_extension:
        Refuse a full-model load when the fused extension is absent.  Set to
        ``False`` only for structural debugging; the fallback is too slow for
        generation at this scale.
    allow_unvalidated:
        Explicit research-only override for packs that have not passed the
        functional quality gate.  Deployment loading is fail-closed by
        default; CRC/parity success alone never promotes a pack.
    """

    init_empty_weights, AutoConfig, AutoModelForCausalLM, _ = _optional_imports()
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if require_cuda_extension and not cuda_backend_available():
        raise RuntimeError(
            "The full 30B edge runtime requires the fused nhdf_edge_cuda extension. "
            "Build it before loading the model."
        )

    root = Path(pack_dir)
    metadata = root / "hf_metadata"
    reader = PackReader(root, verify_crc=verify_crc)
    if bool(reader.manifest.get("partial_pack", False)):
        raise ValueError(
            "cannot load a partial NHDF pack; finish conversion and verify the complete manifest first"
        )
    reader.require_validated(allow_unvalidated=allow_unvalidated)
    runtime_budget: dict[str, float | int] | None = None
    if target.type == "cuda":
        manifest_config = reader.manifest.get("config", {})
        model_profile = manifest_config.get("model", {})
        target_profile = manifest_config.get("target", {})
        pack_bytes = int(reader.manifest.get("summary", {}).get("packed_bytes", 0))
        context_tokens = int(target_profile.get("default_context_tokens", 8192))
        kv_bits = int(target_profile.get("kv_bits", 16))
        elements_per_token = (
            2
            * int(model_profile.get("layers", 0))
            * int(model_profile.get("kv_heads", 0))
            * int(model_profile.get("head_dim", 0))
        )
        if kv_bits == 8:
            kv_group_size = int(target_profile.get("kv_group_size", 64))
            kv_residual_length = int(target_profile.get("kv_residual_length", 128))
            kv_metadata_bits = int(target_profile.get("kv_scale_zero_bits_per_group", 32))
            residual_tokens = min(max(context_tokens - 1, 0), max(kv_residual_length - 1, 0))
            quantized_tokens = context_tokens - residual_tokens
            kv_bytes = int(
                quantized_tokens
                * elements_per_token
                * (kv_bits / 8.0 + kv_metadata_bits / 8.0 / kv_group_size)
                + residual_tokens * elements_per_token * 2.0
            )
        else:
            kv_group_size = 64
            kv_residual_length = 128
            kv_bytes = int(context_tokens * elements_per_token * (kv_bits / 8.0))
        workspace_bytes = int(float(target_profile.get("workspace_gb", 0.0)) * 1e9)
        reserve_bytes = int(float(target_profile.get("runtime_reserve_gb", 0.0)) * 1e9)
        required_bytes = pack_bytes + kv_bytes + workspace_bytes + reserve_bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info(target)
        runtime_budget = {
            "free_bytes_before_load": int(free_bytes),
            "total_device_bytes": int(total_bytes),
            "pack_bytes": pack_bytes,
            "kv_cache_bytes": kv_bytes,
            "workspace_bytes": workspace_bytes,
            "reserve_bytes": reserve_bytes,
            "required_bytes": required_bytes,
            "context_tokens": context_tokens,
            "kv_bits": kv_bits,
        }
        if pack_bytes and free_bytes < required_bytes:
            raise RuntimeError(
                "insufficient free VRAM for the configured NHDF runtime: "
                f"{free_bytes / 1e9:.3f} GB free, {required_bytes / 1e9:.3f} GB required "
                f"for pack + {context_tokens}-token KV cache + workspace + reserve"
            )
    config = AutoConfig.from_pretrained(metadata, local_files_only=True)

    # Rotary-frequency buffers are derived from config and are not present in
    # a checkpoint.  Keeping buffers out of the meta context materializes those
    # small values while parameters remain allocation-free.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    names = set(reader.names())
    expert_groups: dict[str, dict[str, str]] = defaultdict(dict)
    for name in names:
        if name.endswith(".mlp.experts.gate_up_proj") or name.endswith(".mlp.experts.gate_up_proj.weight"):
            prefix = name.rsplit(".gate_up_proj", 1)[0]
            expert_groups[prefix]["gate_up"] = name
        elif name.endswith(".mlp.experts.down_proj") or name.endswith(".mlp.experts.down_proj.weight"):
            prefix = name.rsplit(".down_proj", 1)[0]
            expert_groups[prefix]["down"] = name

    consumed: set[str] = set()
    for expert_path, pair in sorted(expert_groups.items()):
        if set(pair) != {"gate_up", "down"}:
            raise ValueError(f"incomplete expert pair for {expert_path}: {pair}")
        original = _get_path(model, expert_path)
        gate_up = _load_packed(reader, pair["gate_up"], target)
        down = _load_packed(reader, pair["down"], target)
        replacement = NHDFQwen3Experts(
            gate_up,
            down,
            num_experts=int(config.num_experts),
            hidden_dim=int(config.hidden_size),
            intermediate_dim=int(config.moe_intermediate_size),
            activation=original.act_fn,
        )
        _set_path(model, expert_path, replacement)
        consumed.update(pair.values())

    # Replace ordinary matrix modules before assigning one-dimensional raw
    # parameters, because the latter belong to the replacement model tree.
    for name in sorted(names - consumed):
        entry = reader.entry(name)
        mode = entry["metadata"]["policy"]["mode"]
        if mode == "raw" or not name.endswith(".weight"):
            continue
        module_path = name[: -len(".weight")]
        original = _get_path(model, module_path)
        packed = _load_packed(reader, name, target)
        if isinstance(original, nn.Embedding):
            replacement = NHDFPackedEmbedding(packed, padding_idx=original.padding_idx)
        elif isinstance(original, nn.Linear):
            bias = None
            if original.bias is not None and not original.bias.is_meta:
                bias = original.bias
            replacement = NHDFPackedLinear(packed, bias=bias)
        else:
            raise TypeError(f"cannot replace {module_path}: expected Linear/Embedding, got {type(original)!r}")
        _set_path(model, module_path, replacement)
        consumed.add(name)

    # Install raw state words: routers, RMSNorm scales and any future small
    # tensors.  A raw matrix is also assignable, but the default profile does
    # not use one outside the router.
    for name in sorted(names - consumed):
        entry = reader.entry(name)
        mode = entry["metadata"]["policy"]["mode"]
        if mode != "raw":
            raise ValueError(f"unhandled quantized tensor: {name}")
        parameter = _raw_parameter(reader, name, target, dtype)
        _set_path(model, name, parameter)
        consumed.add(name)

    missing = names - consumed
    if missing:
        raise RuntimeError(f"pack entries were not consumed: {sorted(missing)[:20]}")

    # Config-derived buffers (notably RoPE frequencies) start on CPU.  Move
    # them explicitly after replacing all weight-bearing modules; ``model.to``
    # cannot safely be called while diagnosing any remaining meta state.
    for name, buffer in list(model.named_buffers()):
        if not buffer.is_meta and buffer.device != target:
            _set_path(model, name, buffer.to(device=target))

    meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    meta.extend(name for name, buffer in model.named_buffers() if buffer.is_meta)
    if meta:
        raise RuntimeError(
            "model still contains meta parameters or buffers after NHDF replacement: "
            + ", ".join(meta[:20])
        )

    model.eval()
    model.config.use_cache = True
    target_config = reader.manifest.get("config", {}).get("target", {})
    kv_bits = int(target_config.get("kv_bits", 16))
    kv_group_size = int(target_config.get("kv_group_size", 64))
    kv_residual_length = int(target_config.get("kv_residual_length", 128))
    if kv_bits == 8:
        # Transformers' HQQ cache is the available backend that supports the
        # profile's actual 8-bit storage.  Setting generation defaults here
        # prevents a silent fallback to FP16 that would invalidate the VRAM
        # estimate.  HQQ is declared by the runtime extra.
        model.generation_config.cache_implementation = "quantized"
        model.generation_config.cache_config = {
            "backend": "hqq",
            "nbits": 8,
            "axis_key": 0,
            "axis_value": 0,
            "q_group_size": kv_group_size,
            "residual_length": kv_residual_length,
        }
    elif kv_bits != 16:
        raise ValueError(f"unsupported runtime KV cache precision: {kv_bits} bits")
    model.nhdf_runtime_budget = runtime_budget
    return model
